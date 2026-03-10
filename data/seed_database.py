"""Database seeder for NexusAI Support.

Generates 50 realistic customers, 150-300 support tickets, and 100-250 billing
records using the Faker library. Drops and recreates all tables on each run.
"""

import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Add parent directory to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from faker import Faker
from sqlalchemy.orm import Session

from core.database import BillingRecord, Customer, SupportTicket, create_tables, drop_tables, get_session

logger = logging.getLogger(__name__)
fake = Faker()
Faker.seed(42)
random.seed(42)

# ─── Constants ────────────────────────────────────────────────────────────────

PLAN_TYPES = ["Basic", "Pro", "Enterprise"]
PLAN_WEIGHTS = [0.4, 0.35, 0.25]

ACCOUNT_STATUSES = ["Active", "Active", "Active", "Suspended", "Cancelled"]

TICKET_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
TICKET_STATUS_WEIGHTS = [0.25, 0.20, 0.35, 0.20]

TICKET_PRIORITIES = ["Low", "Medium", "High", "Critical"]
TICKET_PRIORITY_WEIGHTS = [0.25, 0.40, 0.25, 0.10]

TICKET_CATEGORIES = ["Billing", "Technical", "General", "Refund", "Account"]
TICKET_CATEGORY_WEIGHTS = [0.25, 0.30, 0.20, 0.15, 0.10]

AGENT_NAMES = [
    "Sarah Mitchell",
    "James Okonkwo",
    "Priya Sharma",
    "Carlos Rivera",
    "Emma Thompson",
    "David Chen",
    "Aisha Williams",
    "Michael O'Brien",
]

BILLING_STATUSES = ["Paid", "Paid", "Paid", "Pending", "Overdue"]

PLAN_AMOUNTS = {
    "Basic": (9.99, 19.99),
    "Pro": (29.99, 79.99),
    "Enterprise": (199.99, 999.99),
}

# ─── Ticket Templates ─────────────────────────────────────────────────────────

BILLING_TICKETS = [
    ("Incorrect charge on invoice #{inv}", "I was charged ${amount} on my last invoice but my plan only costs ${expected}. Please review and issue a correction."),
    ("Double billing for {month}", "I noticed I was billed twice in {month}. The duplicate charge of ${amount} needs to be reversed immediately."),
    ("Invoice not received for {month}", "I have not received my invoice for {month}. Please resend to {email} or update my billing email address."),
    ("Upgrade not reflected in billing", "I upgraded to {plan} plan on {date} but my invoice still shows the old rate. Please adjust my billing."),
    ("Request for billing statement", "I need a detailed billing statement for the past 6 months for accounting purposes. Please provide in PDF format."),
]

TECHNICAL_TICKETS = [
    ("API rate limit errors in production", "Our integration is consistently hitting rate limit errors (429) starting {date}. Error: {error_code}. This is blocking our production workflow."),
    ("Dashboard not loading - 500 error", "The analytics dashboard returns a 500 internal server error since yesterday's update. Affected users: {count}. Browser: Chrome 119."),
    ("Data export feature broken", "The CSV export function in Reports section produces an empty file. This worked fine before {date}. We urgently need this for end-of-month reporting."),
    ("SSO integration failing", "Our SAML SSO configuration stopped working after we updated our identity provider. Getting error: 'Invalid assertion signature'. Reference ID: {error_code}."),
    ("Mobile app crashes on login", "The iOS app version {version} crashes immediately after entering credentials. Reproducible on iPhone 14 and 15 Pro. Crash log attached."),
    ("Webhook not firing on events", "Webhooks for 'order.completed' events stopped firing as of {date}. Our fulfillment system depends on these. Endpoint: {url}"),
]

REFUND_TICKETS = [
    ("Refund request - service downtime", "Your platform was down for {hours} hours on {date} causing significant business impact. Requesting full refund for that billing period per SLA terms."),
    ("Cancelled subscription - refund remaining balance", "I cancelled my annual subscription on {date} with {months} months remaining. Please process the prorated refund of ${amount}."),
    ("Charged after cancellation", "I cancelled my account on {date} but was charged ${amount} on {charge_date}. Please refund immediately and confirm account closure."),
]

ACCOUNT_TICKETS = [
    ("Unable to reset password", "The password reset email is not arriving at {email}. I've checked spam. Please manually reset my account or send to alternate email."),
    ("Account locked after failed logins", "My account was locked after multiple failed login attempts. I'm the legitimate account owner. Please unlock account ID #{account_id}."),
    ("Transfer account to new owner", "Our company was acquired and we need to transfer the account to the new parent company. Please advise on the process."),
    ("GDPR data deletion request", "Per GDPR Article 17, I request complete deletion of all personal data associated with this account. Please confirm within 30 days."),
]

GENERAL_TICKETS = [
    ("Question about plan features", "I'm considering upgrading to {plan} plan. Can you clarify whether {feature} is included, and what the API rate limits are?"),
    ("Integration documentation needed", "We're integrating with {platform}. The documentation at docs.nexus.io is outdated. Can you share current webhook payload examples?"),
    ("Feature request: bulk export", "We manage {count} accounts and need bulk export functionality. Is this on the roadmap? Happy to beta test."),
    ("Onboarding call request", "We just signed up for Enterprise plan. We'd like to schedule an onboarding call to discuss implementation. POC is {name} ({email})."),
]

RESOLUTION_NOTES = [
    "Issue confirmed and resolved. Applied credit to next invoice.",
    "Root cause identified as server-side caching bug. Deployed hotfix v2.1.4. Verified resolution with customer.",
    "Refund of ${amount} processed. Will appear in 3-5 business days.",
    "Escalated to engineering team. Workaround provided: {workaround}",
    "Account settings updated as requested. Customer confirmed resolution.",
    "Documentation updated and shared with customer. Follow-up call scheduled.",
    "Duplicate charge reversed. New invoice sent.",
    "Password reset completed. Security review passed.",
    "Data export repaired in latest release. Customer notified.",
    "SLA credit applied: ${amount}. Incident post-mortem shared.",
]


# ─── Seed Functions ────────────────────────────────────────────────────────────

def generate_customers(session: Session, count: int = 50) -> List[Customer]:
    """Generate and insert synthetic customer records.

    Args:
        session: Active database session.
        count: Number of customers to generate.

    Returns:
        List[Customer]: List of created customer objects.
    """
    customers = []
    emails_used = set()

    for i in range(count):
        email = fake.unique.email()
        while email in emails_used:
            email = fake.unique.email()
        emails_used.add(email)

        plan = random.choices(PLAN_TYPES, weights=PLAN_WEIGHTS)[0]
        status = random.choice(ACCOUNT_STATUSES)
        created = fake.date_time_between(start_date="-3y", end_date="now")

        customer = Customer(
            name=fake.name(),
            email=email,
            phone=fake.phone_number(),
            plan_type=plan,
            account_status=status,
            created_at=created,
            location=f"{fake.city()}, {fake.country()}",
            company_name=fake.company() if random.random() > 0.3 else None,
        )
        session.add(customer)
        customers.append(customer)

    session.flush()
    logger.info("Generated %d customers", len(customers))
    return customers


def _fill_template(template: str) -> str:
    """Fill a ticket description template with fake data.

    Args:
        template: Template string with placeholders.

    Returns:
        str: Filled template string.
    """
    replacements = {
        "{inv}": str(random.randint(1000, 9999)),
        "{amount}": f"{random.uniform(10, 500):.2f}",
        "{expected}": f"{random.uniform(10, 200):.2f}",
        "{month}": fake.month_name() + " " + str(fake.year()),
        "{email}": fake.email(),
        "{plan}": random.choice(PLAN_TYPES),
        "{date}": fake.date_this_year().strftime("%B %d, %Y"),
        "{error_code}": f"ERR_{random.randint(1000, 9999)}",
        "{count}": str(random.randint(5, 500)),
        "{version}": f"{random.randint(2, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
        "{url}": fake.url(),
        "{hours}": str(random.randint(2, 12)),
        "{months}": str(random.randint(1, 11)),
        "{charge_date}": fake.date_this_month().strftime("%B %d, %Y"),
        "{email}": fake.email(),
        "{account_id}": str(random.randint(100000, 999999)),
        "{platform}": random.choice(["Salesforce", "HubSpot", "Zapier", "Slack", "Shopify"]),
        "{feature}": random.choice(["SSO", "API access", "custom reporting", "multi-user seats", "SLA guarantee"]),
        "{name}": fake.name(),
        "{workaround}": "Use the legacy API endpoint /v1/export as temporary measure",
    }
    result = template
    for key, val in replacements.items():
        result = result.replace(key, val)
    return result


def generate_tickets(session: Session, customers: List[Customer]) -> List[SupportTicket]:
    """Generate and insert support tickets for each customer.

    Args:
        session: Active database session.
        customers: List of customer objects to create tickets for.

    Returns:
        List[SupportTicket]: All created ticket objects.
    """
    tickets = []
    category_templates = {
        "Billing": BILLING_TICKETS,
        "Technical": TECHNICAL_TICKETS,
        "Refund": REFUND_TICKETS,
        "Account": ACCOUNT_TICKETS,
        "General": GENERAL_TICKETS,
    }

    for customer in customers:
        ticket_count = random.randint(3, 8)
        for _ in range(ticket_count):
            category = random.choices(TICKET_CATEGORIES, weights=TICKET_CATEGORY_WEIGHTS)[0]
            status = random.choices(TICKET_STATUSES, weights=TICKET_STATUS_WEIGHTS)[0]
            priority = random.choices(TICKET_PRIORITIES, weights=TICKET_PRIORITY_WEIGHTS)[0]

            templates = category_templates[category]
            title_tmpl, desc_tmpl = random.choice(templates)
            title = _fill_template(title_tmpl)[:250]
            description = _fill_template(desc_tmpl)

            created_at = customer.created_at + timedelta(
                days=random.randint(1, max(1, (datetime.utcnow() - customer.created_at).days))
            )

            resolved_at = None
            resolution_notes = None
            agent_name = random.choice(AGENT_NAMES) if status in ("In Progress", "Resolved", "Closed") else None

            if status in ("Resolved", "Closed"):
                resolved_at = created_at + timedelta(hours=random.randint(1, 168))
                note_tmpl = random.choice(RESOLUTION_NOTES)
                resolution_notes = note_tmpl.replace(
                    "${amount}", f"{random.uniform(10, 500):.2f}"
                ).replace("{workaround}", "Use the legacy API endpoint /v1/export as temporary measure")

            ticket = SupportTicket(
                customer_id=customer.id,
                title=title,
                description=description,
                status=status,
                priority=priority,
                category=category,
                created_at=created_at,
                resolved_at=resolved_at,
                resolution_notes=resolution_notes,
                agent_name=agent_name,
            )
            session.add(ticket)
            tickets.append(ticket)

    session.flush()
    logger.info("Generated %d tickets", len(tickets))
    return tickets


def generate_billing(session: Session, customers: List[Customer]) -> List[BillingRecord]:
    """Generate billing records for each customer.

    Args:
        session: Active database session.
        customers: List of customer objects.

    Returns:
        List[BillingRecord]: All created billing record objects.
    """
    records = []
    billing_descriptions = [
        "Monthly subscription fee",
        "Annual plan renewal",
        "Overage charges - API calls",
        "Add-on: Premium support",
        "Add-on: Additional seats",
        "Prorated upgrade adjustment",
        "Professional services",
    ]

    for customer in customers:
        record_count = random.randint(2, 5)
        min_amt, max_amt = PLAN_AMOUNTS[customer.plan_type]

        for i in range(record_count):
            invoice_date = customer.created_at + timedelta(days=30 * i + random.randint(0, 5))
            due_date = invoice_date + timedelta(days=30)
            status = random.choice(BILLING_STATUSES)

            # Older records more likely to be paid
            if invoice_date < datetime.utcnow() - timedelta(days=60):
                status = random.choices(["Paid", "Overdue"], weights=[0.85, 0.15])[0]

            record = BillingRecord(
                customer_id=customer.id,
                amount=round(random.uniform(min_amt, max_amt), 2),
                status=status,
                invoice_date=invoice_date,
                due_date=due_date,
                description=random.choice(billing_descriptions),
            )
            session.add(record)
            records.append(record)

    session.flush()
    logger.info("Generated %d billing records", len(records))
    return records


def print_summary(customers: List[Customer], tickets: List[SupportTicket], billing: List[BillingRecord]) -> None:
    """Print a formatted summary table of seeded data.

    Args:
        customers: List of created customers.
        tickets: List of created tickets.
        billing: List of created billing records.
    """
    print("\n" + "═" * 60)
    print("  NexusAI Support — Database Seed Summary")
    print("═" * 60)

    # Customer breakdown
    plan_counts = {p: sum(1 for c in customers if c.plan_type == p) for p in PLAN_TYPES}
    status_counts = {s: sum(1 for c in customers if c.account_status == s) for s in ["Active", "Suspended", "Cancelled"]}

    print(f"\n  Customers ({len(customers)} total)")
    print(f"  {'Plan':<15} {'Count':<10}")
    print(f"  {'-'*25}")
    for plan, count in plan_counts.items():
        print(f"  {plan:<15} {count:<10}")
    print(f"\n  Active: {status_counts.get('Active', 0)}  |  Suspended: {status_counts.get('Suspended', 0)}  |  Cancelled: {status_counts.get('Cancelled', 0)}")

    # Ticket breakdown
    status_t = {s: sum(1 for t in tickets if t.status == s) for s in TICKET_STATUSES}
    priority_t = {p: sum(1 for t in tickets if t.priority == p) for p in TICKET_PRIORITIES}

    print(f"\n  Tickets ({len(tickets)} total)")
    print(f"  {'Status':<15} {'Count':<10}  {'Priority':<12} {'Count'}")
    print(f"  {'-'*50}")
    for (s, sc), (p, pc) in zip(status_t.items(), priority_t.items()):
        print(f"  {s:<15} {sc:<10}  {p:<12} {pc}")

    # Billing breakdown
    b_status = {s: sum(1 for b in billing if b.status == s) for s in ["Paid", "Pending", "Overdue"]}
    total_outstanding = sum(b.amount for b in billing if b.status in ("Pending", "Overdue"))

    print(f"\n  Billing ({len(billing)} records)")
    for s, count in b_status.items():
        print(f"  {s:<15} {count} records")
    print(f"  Outstanding: ${total_outstanding:,.2f}")
    print("\n" + "═" * 60)


def main() -> None:
    """Main seeding function. Drops all tables, recreates, and seeds with data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("🌱 Starting NexusAI Support database seed...")

    drop_tables()
    create_tables()
    print("✅ Database tables created")

    # Snapshot plain values inside session to avoid DetachedInstanceError
    with get_session() as session:
        customers = generate_customers(session, count=50)
        session.flush()
        print(f"✅ Generated {len(customers)} customers")

        tickets = generate_tickets(session, customers)
        session.flush()
        print(f"✅ Generated {len(tickets)} support tickets")

        billing = generate_billing(session, customers)
        session.flush()
        print(f"✅ Generated {len(billing)} billing records")

        # Read all attributes while session is still open
        customer_plans    = [c.plan_type for c in customers]
        customer_statuses = [c.account_status for c in customers]
        ticket_statuses   = [t.status for t in tickets]
        ticket_priorities = [t.priority for t in tickets]
        billing_statuses  = [b.status for b in billing]
        billing_amounts   = [(b.status, b.amount) for b in billing]
        n_customers, n_tickets, n_billing = len(customers), len(tickets), len(billing)

    # Print summary from plain lists — no ORM session needed
    print("\n" + "═" * 60)
    print("  NexusAI Support — Database Seed Summary")
    print("═" * 60)

    plan_counts   = {p: customer_plans.count(p) for p in PLAN_TYPES}
    status_counts = {s: customer_statuses.count(s) for s in ["Active", "Suspended", "Cancelled"]}
    print(f"\n  Customers ({n_customers} total)")
    print(f"  {'Plan':<15} {'Count':<10}\n  {'-'*25}")
    for plan, count in plan_counts.items():
        print(f"  {plan:<15} {count:<10}")
    print(f"\n  Active: {status_counts.get('Active',0)}  |  Suspended: {status_counts.get('Suspended',0)}  |  Cancelled: {status_counts.get('Cancelled',0)}")

    status_t   = {s: ticket_statuses.count(s) for s in TICKET_STATUSES}
    priority_t = {p: ticket_priorities.count(p) for p in TICKET_PRIORITIES}
    print(f"\n  Tickets ({n_tickets} total)")
    print(f"  {'Status':<15} {'Count':<10}  {'Priority':<12} {'Count'}\n  {'-'*50}")
    for (s, sc), (p, pc) in zip(status_t.items(), priority_t.items()):
        print(f"  {s:<15} {sc:<10}  {p:<12} {pc}")

    b_status    = {s: billing_statuses.count(s) for s in ["Paid", "Pending", "Overdue"]}
    outstanding = sum(amt for st, amt in billing_amounts if st in ("Pending", "Overdue"))
    print(f"\n  Billing ({n_billing} records)")
    for s, count in b_status.items():
        print(f"  {s:<15} {count} records")
    print(f"  Outstanding: ${outstanding:,.2f}\n" + "═" * 60)

    print("🎉 Database seeding complete!\n")


if __name__ == "__main__":
    main()