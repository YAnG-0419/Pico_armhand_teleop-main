class SourceArbiter:
    def __init__(self, allowed_sources, timeout):
        allowed = tuple(str(source) for source in allowed_sources)
        if not allowed or any(not source for source in allowed):
            raise ValueError("At least one non-empty source must be allowed.")
        if timeout <= 0:
            raise ValueError("Source timeout must be positive.")
        self.allowed_sources = frozenset(allowed)
        self.timeout = float(timeout)
        self.source = None
        self.session_id = None
        self.sequence = -1
        self.received_at = None

    def reset(self):
        self.source = None
        self.session_id = None
        self.sequence = -1
        self.received_at = None

    def accept(self, source, session_id, sequence, now):
        if source not in self.allowed_sources:
            raise ValueError(f"Teleop source is not allowed: {source}")
        if not session_id or sequence < 0:
            raise ValueError("Command session and sequence are invalid.")
        expired = self.received_at is None or now - self.received_at > self.timeout
        changed = source != self.source or session_id != self.session_id
        if changed and not expired:
            raise ValueError(
                f"Source {self.source} still owns the command channel."
            )
        new_session = changed
        if new_session:
            self.source = source
            self.session_id = session_id
            self.sequence = -1
        if sequence <= self.sequence:
            raise ValueError("Command sequence is stale or duplicated.")
        self.sequence = sequence
        self.received_at = now
        return new_session
