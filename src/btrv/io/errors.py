class ImportValidationError(ValueError):
    """Raised for malformed or misaligned external data; the message is user-actionable."""


class ExportValidationError(ValueError):
    """Raised when an export cannot be produced correctly (never silently fixed)."""
