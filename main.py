from fastapi import FastAPI
from fastapi.responses import JSONResponse

import hashlib
import json
import math


app = FastAPI()

SAFE_INT_MAX = 9007199254740991

DAG = [
    "verify_data",
    "prepare",
    "train",
    "evaluate",
    "register",
    "publish",
]

NODE_PARENT = {
    "verify_data": None,
    "prepare": "verify_data",
    "train": "prepare",
    "evaluate": "train",
    "register": "evaluate",
    "publish": "register",
}

# session -> state
SESSIONS = {}


# ============================================================
# HELPERS
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":")
    )


def sha256_json_array(values):
    raw = compact_json(values).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def utf8_key(value):
    return value.encode("utf-8")


def unique_sorted(codes):
    return sorted(set(codes), key=utf8_key)


def safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def positive_safe_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= SAFE_INT_MAX
    )


def nonempty_string(value):
    return isinstance(value, str) and value != ""


def digest_like(value):
    return isinstance(value, str) and value != ""


def clone(value):
    return json.loads(json.dumps(value))


# ============================================================
# REQUEST VALIDATION
# ============================================================

INPUT_FIELDS = [
    "generation",
    "checksum",
    "canonicalData",
    "prepareCode",
    "prepareConfig",
    "trainCode",
    "trainConfig",
    "runtime",
    "evaluateCode",
    "evaluateConfig",
    "schemaDigest",
    "publishConfig",
]


EVENT_FIELDS = {
    "eventId",
    "revision",
    "node",
    "attempt",
    "status",
    "key",
    "artifactDigest",
    "receiptId",
}

VALID_STATUSES = {
    "started",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
}


def validate_request_shape(payload):
    if not isinstance(payload, dict):
        return False

    required = {
        "session",
        "revision",
        "inputs",
        "events",
    }

    if not required.issubset(payload.keys()):
        return False

    if not nonempty_string(payload["session"]):
        return False

    if not positive_safe_int(payload["revision"]):
        return False

    if not isinstance(payload["inputs"], dict):
        return False

    if not isinstance(payload["events"], list):
        return False

    for field in INPUT_FIELDS:
        if field not in payload["inputs"]:
            return False
        if not nonempty_string(payload["inputs"][field]):
            return False

    return True


# ============================================================
# KEY COMPUTATION
# ============================================================

def compute_keys(inputs, artifacts):
    """
    artifacts contains successful upstream artifact digests.
    Returns keys for every DAG node.
    """

    keys = {}

    # verify_data
    keys["verify_data"] = sha256_json_array([
        inputs["generation"],
        inputs["checksum"],
    ])

    # prepare
    if artifacts.get("verify_data") is not None:
        keys["prepare"] = sha256_json_array([
            inputs["canonicalData"],
            inputs["prepareCode"],
            inputs["prepareConfig"],
        ])
    else:
        keys["prepare"] = None

    # train
    if artifacts.get("prepare") is not None:
        keys["train"] = sha256_json_array([
            artifacts["prepare"],
            inputs["trainCode"],
            inputs["trainConfig"],
            inputs["runtime"],
        ])
    else:
        keys["train"] = None

    # evaluate
    if artifacts.get("train") is not None:
        keys["evaluate"] = sha256_json_array([
            artifacts["train"],
            inputs["canonicalData"],
            inputs["evaluateCode"],
            inputs["evaluateConfig"],
        ])
    else:
        keys["evaluate"] = None

    # register
    if artifacts.get("evaluate") is not None:
        keys["register"] = sha256_json_array([
            artifacts["evaluate"],
            inputs["schemaDigest"],
        ])
    else:
        keys["register"] = None

    # publish
    if artifacts.get("register") is not None:
        keys["publish"] = sha256_json_array([
            artifacts["register"],
            inputs["publishConfig"],
        ])
    else:
        keys["publish"] = None

    return keys


# ============================================================
# EVENT VALIDATION
# ============================================================

def valid_event(event):
    if not isinstance(event, dict):
        return False

    if set(event.keys()) != EVENT_FIELDS:
        return False

    if not nonempty_string(event["eventId"]):
        return False

    if not positive_safe_int(event["revision"]):
        return False

    if event["node"] not in DAG:
        return False

    if not positive_safe_int(event["attempt"]):
        return False

    if event["status"] not in VALID_STATUSES:
        return False

    if not nonempty_string(event["key"]):
        return False

    artifact = event["artifactDigest"]
    receipt = event["receiptId"]

    # Success requires artifact.
    if event["status"] == "succeeded":
        if not nonempty_string(artifact):
            return False
    else:
        if artifact is not None:
            return False

    # Register/publish success requires receipt.
    if (
        event["node"] in {"register", "publish"}
        and event["status"] == "succeeded"
    ):
        if not nonempty_string(receipt):
            return False
    else:
        if receipt is not None:
            return False

    return True


# ============================================================
# STATE
# ============================================================

def new_session_state(revision, inputs):
    return {
        "revision": revision,
        "inputs": clone(inputs),

        # Permanently reusable successful content-addressed
        # cache: node -> key -> evidence
        "cache": {
            node: {}
            for node in DAG
        },

        # Current revision execution state:
        # node -> {
        #   "key": key,
        #   "status": ...,
        #   "attempt": n,
        #   "artifactDigest": ...,
        #   "receiptId": ...,
        #   "eventId": ...
        # }
        "current": {},

        # eventId -> canonical event object
        "events": {},
    }


# ============================================================
# INPUT / REVISION HANDLING
# ============================================================

def get_or_create_session(session, revision, inputs):
    state = SESSIONS.get(session)

    if state is None:
        state = new_session_state(
            revision,
            inputs
        )
        SESSIONS[session] = state
        return state, None

    # Newer revision replaces active state but keeps cache.
    if revision > state["revision"]:

        cache = state["cache"]

        new_state = new_session_state(
            revision,
            inputs
        )

        new_state["cache"] = cache

        SESSIONS[session] = new_state

        return new_state, None

    # Older revision events/input are handled separately.
    if revision < state["revision"]:
        return state, "OLDER_REVISION"

    # Same revision must have identical inputs, including extra metadata.
    if state["inputs"] != inputs:
        return state, "REVISION_CONFLICT"

    return state, None


# ============================================================
# CACHE / READINESS
# ============================================================

def reusable_artifacts(state):
    """
    Reconstruct currently reusable artifacts from current keys.
    """

    artifacts = {}

    for node in DAG:

        current = state["current"].get(node)

        if current is None:
            continue

        if current["status"] == "succeeded":
            artifacts[node] = current["artifactDigest"]

    return artifacts


def get_node_dependencies(node, inputs, artifacts):
    if node == "verify_data":
        return {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
        }

    if node == "prepare":
        return {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
        }

    if node == "train":
        return {
            "prepareArtifact": artifacts.get("prepare"),
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
        }

    if node == "evaluate":
        return {
            "trainArtifact": artifacts.get("train"),
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
        }

    if node == "register":
        return {
            "evaluateArtifact": artifacts.get("evaluate"),
            "schemaDigest": inputs["schemaDigest"],
        }

    if node == "publish":
        return {
            "registerArtifact": artifacts.get("register"),
            "publishConfig": inputs["publishConfig"],
        }

    return {}


def node_key(node, inputs, artifacts):
    deps = get_node_dependencies(
        node,
        inputs,
        artifacts
    )

    if node == "verify_data":
        values = [
            deps["generation"],
            deps["checksum"],
        ]
    elif node == "prepare":
        values = [
            deps["canonicalData"],
            deps["prepareCode"],
            deps["prepareConfig"],
        ]
    elif node == "train":
        if deps["prepareArtifact"] is None:
            return None
        values = [
            deps["prepareArtifact"],
            deps["trainCode"],
            deps["trainConfig"],
            deps["runtime"],
        ]
    elif node == "evaluate":
        if deps["trainArtifact"] is None:
            return None
        values = [
            deps["trainArtifact"],
            deps["canonicalData"],
            deps["evaluateCode"],
            deps["evaluateConfig"],
        ]
    elif node == "register":
        if deps["evaluateArtifact"] is None:
            return None
        values = [
            deps["evaluateArtifact"],
            deps["schemaDigest"],
        ]
    elif node == "publish":
        if deps["registerArtifact"] is None:
            return None
        values = [
            deps["registerArtifact"],
            deps["publishConfig"],
        ]
    else:
        return None

    return sha256_json_array(values)


def parent_ready(node, state):
    parent = NODE_PARENT[node]

    if parent is None:
        return True

    parent_state = state["current"].get(parent)

    if parent_state is None:
        return False

    return parent_state["status"] == "succeeded"


# ============================================================
# EVENT PROCESSING
# ============================================================

def process_event(state, event, accepted_ids, ignored_ids):
    event_id = event["eventId"]

    # Exact replay is ignored.
    existing = state["events"].get(event_id)

    if existing is not None:

        if existing == event:
            ignored_ids.append(event_id)
            return None

        return "EVENT_ID_CONFLICT"

    current_revision = state["revision"]

    # Older revision events are ignored.
    if event["revision"] != current_revision:
        ignored_ids.append(event_id)
        return None

    node = event["node"]

    # Compute key from current reusable parents.
    artifacts = reusable_artifacts(state)

    expected_key = node_key(
        node,
        state["inputs"],
        artifacts
    )

    # Unavailable parent / wrong key => ignore.
    if expected_key is None:
        ignored_ids.append(event_id)
        return None

    if event["key"] != expected_key:
        ignored_ids.append(event_id)
        return None

    # Save exact event only after all ignore/conflict checks pass.
    current = state["current"].get(node)

    # ========================================================
    # succeeded/current cache state
    # ========================================================

    if current is not None:

        status = current["status"]
        attempt = current["attempt"]

        if status == "succeeded":

            if event["status"] == "succeeded":

                if (
                    event["artifactDigest"]
                    != current["artifactDigest"]
                ):
                    return "EVIDENCE_CONFLICT"

                ignored_ids.append(event_id)
                return None

            # Any other new event conflicts.
            return "STATUS_CONFLICT"

        if status == "terminal_failed":
            return "STATUS_CONFLICT"

        # ----------------------------------------------------
        # Started / retryable failed
        # ----------------------------------------------------

        if status == "started":

            if (
                event["status"]
                in {
                    "succeeded",
                    "retryable_failed",
                    "terminal_failed",
                }
                and event["attempt"] == attempt
            ):

                pass

            else:

                if event["attempt"] < attempt:
                    ignored_ids.append(event_id)
                    return None

                return "STATUS_CONFLICT"

        elif status == "retryable_failed":

            if (
                event["status"] == "started"
                and event["attempt"] == attempt + 1
            ):
                pass

            elif event["attempt"] < attempt:
                ignored_ids.append(event_id)
                return None

            else:
                return "STATUS_CONFLICT"

    else:

        # No current state.
        # Only started attempt 1 is accepted.
        if not (
            event["status"] == "started"
            and event["attempt"] == 1
        ):
            ignored_ids.append(event_id)
            return None

    # ========================================================
    # Apply event
    # ========================================================

    new_state = {
        "key": event["key"],
        "status": event["status"],
        "attempt": event["attempt"],
        "artifactDigest": (
            event["artifactDigest"]
            if event["status"] == "succeeded"
            else None
        ),
        "receiptId": (
            event["receiptId"]
            if event["status"] == "succeeded"
            else None
        ),
        "eventId": event_id,
    }

    state["current"][node] = new_state
    state["events"][event_id] = clone(event)

    # Permanent successful cache binding.
    if event["status"] == "succeeded":

        cache_entry = state["cache"][node].get(
            event["key"]
        )

        if cache_entry is None:

            state["cache"][node][event["key"]] = {
                "artifactDigest": event["artifactDigest"],
                "eventId": event_id,
                "receiptId": event["receiptId"],
            }

        else:

            if (
                cache_entry["artifactDigest"]
                != event["artifactDigest"]
            ):
                return "EVIDENCE_CONFLICT"

    accepted_ids.append(event_id)

    return None


# ============================================================
# RESPONSE GENERATION
# ============================================================

def node_response(state, node):
    inputs = state["inputs"]

    artifacts = reusable_artifacts(state)

    key = node_key(
        node,
        inputs,
        artifacts
    )

    dependency_digests = (
        get_node_dependencies(
            node,
            inputs,
            artifacts
        )
    )

    dependency_digests = {
        k: v
        for k, v in dependency_digests.items()
        if v is not None
    }

    dependency_digests["cacheKey"] = key

    current = state["current"].get(node)

    cache_entry = None

    if key is not None:
        cache_entry = state["cache"][node].get(key)

    # Cached immutable success.
    if cache_entry is not None:

        return {
            "node": node,
            "action": "reuse",
            "reasonCodes": ["CACHE_HIT"],
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": [
                cache_entry["eventId"]
            ],
        }

    # Current state
    if current is not None:

        if current["status"] == "started":

            return {
                "node": node,
                "action": "block",
                "reasonCodes": ["RUNNING"],
                "dependencyDigests": dependency_digests,
                "triggeringEventIds": [
                    current["eventId"]
                ],
            }

        if current["status"] == "terminal_failed":

            return {
                "node": node,
                "action": "block",
                "reasonCodes": ["TERMINAL_FAILURE"],
                "dependencyDigests": dependency_digests,
                "triggeringEventIds": [
                    current["eventId"]
                ],
            }

        if current["status"] == "retryable_failed":

            return {
                "node": node,
                "action": "rerun",
                "reasonCodes": [
                    "RETRYABLE_FAILURE"
                ],
                "dependencyDigests": dependency_digests,
                "triggeringEventIds": [
                    current["eventId"]
                ],
            }

    parent = NODE_PARENT[node]

    if parent is not None:

        parent_result = node_response(
            state,
            parent
        )

        if parent_result["action"] == "block":

            if (
                "TERMINAL_FAILURE"
                in parent_result["reasonCodes"]
                or "UPSTREAM_TERMINAL"
                in parent_result["reasonCodes"]
            ):

                return {
                    "node": node,
                    "action": "block",
                    "reasonCodes": [
                        "UPSTREAM_TERMINAL"
                    ],
                    "dependencyDigests":
                        dependency_digests,
                    "triggeringEventIds":
                        parent_result[
                            "triggeringEventIds"
                        ],
                }

            return {
                "node": node,
                "action": "block",
                "reasonCodes": [
                    "UPSTREAM_PENDING"
                ],
                "dependencyDigests":
                    dependency_digests,
                "triggeringEventIds":
                    parent_result[
                        "triggeringEventIds"
                    ],
            }

    # No cache and ready.
    return {
        "node": node,
        "action": "rerun",
        "reasonCodes": ["CACHE_MISS"],
        "dependencyDigests": dependency_digests,
        "triggeringEventIds": [],
    }


def build_response(state, accepted_ids, ignored_ids):
    return {
        "revision": state["revision"],
        "acceptedEventIds": accepted_ids,
        "ignoredEventIds": ignored_ids,
        "nodes": [
            node_response(state, node)
            for node in DAG
        ],
    }


# ============================================================
# ENDPOINT
# ============================================================

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/pipeline")
async def pipeline(payload: dict):

    if not validate_request_shape(payload):
        return JSONResponse(
            status_code=400,
            content={
                "error": "INVALID_REQUEST"
            }
        )

    session = payload["session"]
    revision = payload["revision"]
    inputs = payload["inputs"]
    events = payload["events"]

    # --------------------------------------------------------
    # Session / revision handling
    # --------------------------------------------------------

    state, issue = get_or_create_session(
        session,
        revision,
        inputs
    )

    if issue == "REVISION_CONFLICT":

        return JSONResponse(
            status_code=409,
            content={
                "error": "REVISION_CONFLICT"
            }
        )

    # --------------------------------------------------------
    # Validate every event before mutating state.
    # Invalid event => atomic batch rejection.
    # --------------------------------------------------------

    for event in events:

        if not valid_event(event):

            return JSONResponse(
                status_code=409,
                content={
                    "error": "INVALID_EVENT"
                }
            )

    # --------------------------------------------------------
    # Work on a deep copy so a conflict rolls back
    # the entire batch.
    # --------------------------------------------------------

    working = clone(state)

    accepted_ids = []
    ignored_ids = []

    for event in events:

        error = process_event(
            working,
            event,
            accepted_ids,
            ignored_ids
        )

        if error is not None:

            return JSONResponse(
                status_code=409,
                content={
                    "error": error
                }
            )

    # --------------------------------------------------------
    # Commit atomically.
    # --------------------------------------------------------

    SESSIONS[session] = working

    return build_response(
        working,
        accepted_ids,
        ignored_ids
    )
