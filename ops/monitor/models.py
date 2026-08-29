from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass
class Finding:
    check: str
    severity: Severity
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class HealthReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity == Severity.ERROR for f in self.findings)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warn_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARN)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def merge(self, other: HealthReport) -> None:
        self.findings.extend(other.findings)
