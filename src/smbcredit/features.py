"""Feature lists, monotone constraints and human-readable reason-code text."""

# LEAKAGE NOTE: `term_months` is excluded. In this FOIA extract the term of charged-off loans
# appears to reflect post-default modification (non-standard terms such as 42 or 46 months default at
# 80%+, standard 84-month terms at <1%). It is reported as a leakage diagnostic only.
LEAKY = ["term_months"]

NUMERIC = [
    "log_approved_amt", "sba_guarantee_pct", "log_employees",
    "is_new_business", "is_franchise", "is_revolver", "is_lowdoc",
    "is_out_of_state_lender", "jobs_created", "jobs_retained",
    "days_to_disburse", "is_rural", "rel_bank_chgoff", "log_bank_prior_loans",
    "rel_naics_chgoff", "rel_state_chgoff",
]
CATEGORICAL = ["naics2"]
FEATURES = NUMERIC + CATEGORICAL

# Economically-motivated monotone constraints (+1: risk rises with x, -1: risk falls).
# Only where the direction is defensible to a credit committee; everything else is free.
MONOTONE = {
    "log_employees": -1,          # scale / operating history
    "rel_bank_chgoff": +1,   # lender's observed historical loss experience
    "rel_naics_chgoff": +1,
    "rel_state_chgoff": +1,
}

# Reason-code text is value-aware: binaries state which side of the flag the applicant is on,
# continuous features state high/low versus the training median.
BINARY_TEXT = {
    "is_new_business": ("New business (< 2 years operating)", "Established business"),
    "is_franchise": ("Franchise borrower", "Non-franchise borrower"),
    "is_revolver": ("Revolving line of credit", "Term (non-revolving) structure"),
    "is_lowdoc": ("Low-documentation program", "Full-documentation program"),
    "is_out_of_state_lender": ("Out-of-state lender", "In-state lender"),
    "is_rural": ("Rural location", "Urban location"),
}
LABELS = {
    "log_approved_amt": "loan size",
    "sba_guarantee_pct": "SBA guarantee share",
    "log_employees": "employee count",
    "jobs_created": "jobs created",
    "jobs_retained": "jobs retained",
    "days_to_disburse": "days from approval to disbursement",
    "rel_bank_chgoff": "lender's historical charge-off rate vs network",
    "log_bank_prior_loans": "lender's SBA track record (prior loans)",
    "rel_naics_chgoff": "industry historical charge-off rate vs network",
    "rel_state_chgoff": "state historical charge-off rate vs network",
    "naics2": "industry sector",
}


def reason_text(feat: str, value, median) -> str:
    if feat in BINARY_TEXT:
        if value is None or value != value:  # NaN
            return f"{feat} missing"
        return BINARY_TEXT[feat][0 if value >= 0.5 else 1]
    if feat == "naics2":
        return f"Industry sector NAICS {value}"
    if value is None or value != value:
        return f"{LABELS[feat]} missing"
    return f"{'High' if value > median else 'Low'} {LABELS[feat]}"
