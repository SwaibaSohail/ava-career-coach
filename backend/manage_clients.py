"""Admin CLI for token metering: client accounts, plans and monthly usage.

Run from backend/ with the virtualenv active:
    python manage_clients.py add "Acme Ltd" --plan starter   # prints the key once
    python manage_clients.py list
    python manage_clients.py set-plan <client_id> pro
    python manage_clients.py plans
    python manage_clients.py usage [--month 2026-10] [--client <client_id>]
Months are calendar months in UTC.
"""

import argparse
import sys

import metering


def _fmt_limit(limit):
    return "uncapped" if limit is None else f"{limit:,}"


def _fmt_cost(client):
    # The report gives null cost only when unpriced calls leave it unknown.
    return "n/a" if client["cost_usd"] is None else f"${client['cost_usd']:.4f}"


def _fmt_calls(client):
    # Calls without a price, and replies cut off mid-stream whose tokens are estimates.
    notes = [f"{n} {kind} call{'s' if n != 1 else ''}"
             for kind, n in (("unpriced", client["unpriced_calls"]), ("estimated", client["estimated_calls"])) if n]
    return f" ({', '.join(notes)})" if notes else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manage_clients.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="create a client and print its key (shown once)")
    add.add_argument("name")
    add.add_argument("--plan", required=True)
    sub.add_parser("list", help="list clients")
    sp = sub.add_parser("set-plan", help="change a client's plan")
    sp.add_argument("client_id")
    sp.add_argument("plan")
    sub.add_parser("plans", help="list plans and their monthly allowances")
    us = sub.add_parser("usage", help="monthly usage per client (UTC month)")
    us.add_argument("--month", help="YYYY-MM; defaults to the current UTC month")
    us.add_argument("--client", help="only this client_id")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "add":
            client_id, key = metering.add_client(args.name, args.plan)
            print(f"Created client {client_id} on plan {args.plan}.")
            print(f"Client key (shown once; store it safely): {key}")
        elif args.cmd == "list":
            for c in metering.list_clients():
                print(f"{c['client_id']:<32} {c['plan']:<10} {c['name']}")
        elif args.cmd == "set-plan":
            metering.set_plan(args.client_id, args.plan)
            print(f"{args.client_id} is now on plan {args.plan}.")
        elif args.cmd == "plans":
            for name, plan in metering.plans().items():
                print(f"{name:<10} {_fmt_limit(plan.get('monthly_tokens')):>12} tokens/month")
        elif args.cmd == "usage":
            report = metering.usage_report(month=args.month, client_id=args.client)
            print(f"Usage for {report['month']} (UTC)")
            for c in report["clients"]:
                # Usage under a client id with no account row has no plan.
                print(f"{c['client_id']:<32} {c['plan'] or '(none)':<10} "
                      f"{c['used_tokens']:>12,} / {_fmt_limit(c['monthly_tokens']):<12} "
                      f"est. cost {_fmt_cost(c)}{_fmt_calls(c)}")
            if not report["clients"]:
                print("No matching clients.")
            print(report["note"])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
