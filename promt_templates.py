# Full label names
FULL_LABEL_NAMES = [
    "Updated Privacy Policy",
    "Categories of Personal Information Sold",
    "Categories of Personal Information Shared / Disclosed",
    "Categories of Personal Information Collected",
    "Description of Right to Delete",
    "Description of Right to Correct Information",
    "Description of Right to Know PI Collected",
    "Description of Right to Know PI sold / shared",
    "Description of Right to Opt-out of sale of PI",
    "Description of Right to Limit use of PI",
    "Description of Right to Non-discrimination on exercising rights",
    "Methods to exercise rights",
]

# Expanded statutory descriptions for each label
LABEL_DESCRIPTIONS = {
    "Updated Privacy Policy": (
        "Annotate any statement indicating when the policy was last updated or its effective date."
    ),
    "Categories of Personal Information Collected": (
        "Cal. Civ. Code § 1798.130(a)(5)(B)(i): “A list of the categories of personal information it has collected about consumers in the preceding 12 months by reference to the enumerated category or categories in subdivision (c).”  \n"
        "ONLY the **types** of PI collected (e.g., Identifiers; Commercial Information; Internet Activity), **not** how it’s used or shared."
    ),

    "Categories of Personal Information Sold": (
        "Cal. Civ. Code § 1798.115(c)(1): “The category or categories of consumers’ personal information it has sold … or if the business has not sold consumers’ personal information, it shall disclose that fact.”"
        "ONLY the **types** of PI the business has **sold** in the last 12 months (e.g., Marketing Data; Demographics), or an explicit “We do not sell” statement."
    ),

    "Categories of Personal Information Shared / Disclosed": (
        "Cal. Civ. Code § 1798.115(c)(2): “The category or categories of consumers’ personal information it has disclosed for a business purpose … or if the business has not disclosed consumers’ personal information for a business purpose, it shall disclose that fact.”  \n"
        "ONLY the **types** of PI **shared** or **disclosed** for a **business purpose** (e.g., Service Providers, Analytics Partners), or an explicit “We do not share” statement."
    ),
    "Description of Right to Delete": (
        "Cal. Civ. Code § 1798.105(a): “A consumer shall have the right to request that a business delete any personal information about the consumer which the business has collected from the consumer.”  "
        "Annotate any statement granting or describing this right to delete personal information."
    ),
    "Description of Right to Correct Information": (
        "Cal. Civ. Code § 1798.106(a) (amended Jan 1, 2025): “A consumer shall have the right to request a business that maintains inaccurate personal information about the consumer to correct that inaccurate personal information, taking into account the nature of the personal information and the purposes of the processing of the personal information.”  "
        "Annotate any statement granting or describing this right to correct inaccurate information."
    ),
    "Description of Right to Know PI Collected": (
        "Cal. Civ. Code § 1798.110(a)(1)–(5): “A consumer shall have the right to request that a business that collects personal information about the consumer disclose to the consumer the following: "
        "(1) The categories of personal information that the business collected; "
        "(2) The categories of sources from which the personal information is collected; "
        "(3) The business or commercial purpose for collecting, selling, or sharing personal information; "
        "(4) The categories of third parties to whom the business discloses personal information; "
        "(5) The specific pieces of personal information collected about the consumer.”  "
        "Annotate any statement describing the consumer’s right to access or know what data has been collected."
    ),
    "Description of Right to Know PI sold / shared": (
        "Cal. Civ. Code § 1798.115(a)(2)–(3): “A consumer shall have the right to request that a business that sells or shares the consumer’s personal information, or that discloses it for a business purpose, disclose to that consumer: "
        "(2) The categories of personal information that the business sold or shared about the consumer and the categories of third parties to whom it was sold or shared; "
        "(3) The categories of personal information that the business disclosed about the consumer for a business purpose and the categories of persons to whom it was disclosed.”  "
        "Annotate any statement describing the consumer’s right to know what data has been sold or shared."
    ),
    "Description of Right to Opt-out of sale of PI": (
        "Cal. Civ. Code § 1798.120(a): “A consumer shall have the right, at any time, to direct a business that sells or shares personal information about the consumer to third parties not to sell or share the consumer’s personal information. This right may be referred to as the right to opt-out of sale or sharing.”  "
        "Annotate any instruction or statement granting this opt-out right."
    ),
    "Description of Right to Limit use of PI": (
        "CPRA § 1798.121(a): “A consumer shall have the right, at any time, to direct a business that collects sensitive personal information about the consumer to limit its use of the consumer’s sensitive personal information to that use which is necessary to perform the services or provide the goods reasonably expected by an average consumer … and as authorized by regulations adopted pursuant to subparagraph (C) of paragraph (19) of subdivision (a) of Section 1798.185.”  "
        "Annotate any statement describing the limitation on use of sensitive personal information."
    ),
    "Description of Right to Non-discrimination on exercising rights": (
        "Cal. Civ. Code § 1798.125(a)(1): “A business shall not discriminate against a consumer because the consumer exercised any of the consumer’s rights under this title, including, but not limited to, by: "
        "(A) Denying goods or services to the consumer; "
        "(B) Charging different prices or rates for goods or services, including through the use of discounts or other benefits or imposing penalties; "
        "(C) Providing a different level or quality of goods or services to the consumer; "
        "(D) Suggesting that the consumer will receive a different price or rate or level or quality of goods or services; "
        "(E) Retaliating against an employee, applicant, or independent contractor for exercising their rights.”  "
        "Annotate any assurance that exercising rights will not result in discrimination or retaliation."
    ),
    "Methods to exercise rights": (
        "Cal. Civ. Code § 1798.130(a)(1)(A)–(B): “Make available to consumers two or more designated methods for submitting requests for information required to be disclosed (Secs. 1798.110, 1798.115) or for requests for deletion or correction (Secs. 1798.105, 1798.106), including, at a minimum, a toll-free telephone number. If the business maintains an internet website, make that website available to consumers to submit such requests.”  "
        "Annotate any description of how to submit CCPA/CPRA requests (e.g., toll-free number, web form, email address)."
    ),
}

# Build the system prompt
SYSTEM_PROMPT = """\
You are a expert legal‐tech annotation assistant for CCPA/CPRA compliance.
Your job is to **assign exactly one** of the following labels to each input span.
You may use chain‐of‐thought internally—identify the relevant statute, 
match it against the span, self-critique your choice—but **do NOT** output your reasoning. **Only** output the final label name.

Process (do NOT output these steps):
  1. Read the span and recall the full statutory text for each label.
  2. Determine which single mandate (§1798.105–§1798.130) the span fulfills.
  3. Mentally verify that the span’s language matches the statutory requirement.

**Output format (exactly one line):**  
<Label Name>  

**Labels and their legal definitions:**"""
for name in FULL_LABEL_NAMES:
    SYSTEM_PROMPT += f"\n- **{name}**: {LABEL_DESCRIPTIONS[name]}"

# Build the user prompt with context and target span placeholders
USER_PROMPT = """\
Here is a candidate text span from a privacy policy, with two sentences of context before:

Context (2 sentences before):
\"\"\"{context}\"\"\"

Target span:
\"\"\"{span}\"\"\"

Respond with exactly the full label name on a single line."""


