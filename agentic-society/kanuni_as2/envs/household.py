"""Household money: income, prices, obligations and debt, kept by code.

The model only makes choices (how to spend, whether to work, whether to
borrow). This module turns them into shillings using the livelihood
profiles, so an agent cannot spend money it does not have and every missed
payment is recorded. One row per agent per simulated day goes to
household_days.jsonl.

Figures come from shared/countries/<cc>/livelihood_profiles.json; most are
marked "assumed" there and must be checked before reportable runs.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from agentsociety2.env import EnvBase, tool

from kanuni_as2.common import (World, append_jsonl, check_agent, in_range,
                               stable_rng, value)

SPENDING_PLANS = {"normal": 1.0, "cut_back": 0.8, "cheaper_goods": 0.9}
WORK_PLANS = ("normal", "work_extra", "take_day_off")
EXTRA_WORK_INCOME = 1.25
# Assumed shares used to apply price levers to a household's spending.
BREAD_SHARE_OF_FOOD = 0.10
OTHER_GOODS_SHARE_OF_FOOD = 0.15
MOBILE_MONEY_FEE_SHARE = 0.005
SALARY_DEDUCTIONS = 0.25  # PAYE + housing levy + health + NSSF, assumed
# Loan terms per source: (monthly interest, max loan as months of income).
LOAN_TERMS = {"mobile_loan": (0.075, 0.5), "chama": (0.10, 1.0),
              "family": (0.0, 0.3), "shylock": (0.20, 1.0)}
HAS_CHILDREN_IN_SCHOOL = 0.6   # assumed share of adults paying fees
RENTS_BIKE = 0.7               # assumed share of riders not owning the bike


class KanuniHouseholdEnv(EnvBase):
    """Each person's household budget: cash, income from work, daily costs,
    rent and fees with due dates, and loans."""

    def __init__(self, world: World, log_path: str):
        super().__init__()
        self.world = world
        self.log_path = log_path
        self.civic = None  # set by the runner; tells us who protested
        self.cal = world.econ["calendar"]
        self.state: dict[int, dict] = {}
        self.levers = world.scenario.get("levers", [])

    @classmethod
    def description(cls) -> str:
        return ("Household budget: cash, income, costs, rent and school fees, "
                "and loans. Tools to change spending, work and borrowing.")

    async def init(self, start_datetime: datetime):
        await super().init(start_datetime)
        for aid in self.world.personas:
            self.state[aid] = self._initial_state(aid)

    # ---- setup ----

    def _initial_state(self, aid: int) -> dict:
        prof = self.world.profile(aid)
        pos = self.world.band_position(aid)
        rng = stable_rng(self.world.seed, "household", aid)
        obligations = []
        for ob in prof["obligations"]:
            if ob["name"] == "bike_payment" and rng.random() > RENTS_BIKE:
                continue
            if ob["name"] == "school_fees" and \
                    rng.random() > HAS_CHILDREN_IN_SCHOOL:
                continue
            obligations.append({**ob, "kes": round(in_range(
                value(ob["amount"]), pos))})
        st = {"profile_id": prof["profile_id"], "pos": pos,
              "obligations": obligations, "cash": 0.0, "debt": 0.0,
              "arrears": {}, "spending_plan": "normal",
              "work_plan": "normal", "loans_today": [], "notes": [],
              "last_day": None}
        st["cash"] = round(self._daily_cost_estimate(st) * in_range(
            value(prof["buffers"]["savings_days"]), pos), 0)
        return st

    def _daily_cost_estimate(self, st: dict) -> float:
        per_day = {"daily": 1, "working_day": 6 / 7, "weekly": 1 / 7,
                   "monthly": 1 / 30, "term": 3 / 365, "season": 2 / 365,
                   "yearly": 1 / 365}
        return sum(o["kes"] * per_day.get(o["frequency"], 0)
                   for o in st["obligations"])

    # ---- prices ----

    def _multiplier(self, applies_to: str, day: date) -> float:
        m = 1.0
        for lv in self.levers:
            if lv.get("applies_to") == applies_to and "multiplier" in lv \
                    and day >= date.fromisoformat(lv["effective"]):
                m *= lv["multiplier"]
        return m

    def _food_factor(self, day: date) -> float:
        return 1 + BREAD_SHARE_OF_FOOD * (self._multiplier("bread", day) - 1)

    # ---- income ----

    def _income(self, aid: int, st: dict, day: date, protested: bool) -> float:
        inc = self.world.profile(aid)["income"]
        pos = st["pos"]
        rng = stable_rng(self.world.seed, "income", aid, day)
        pattern = inc["pattern"]
        work = st["work_plan"]
        if pattern == "daily":
            wd = value(inc.get("working_days_per_week", 6))
            days = in_range(wd, pos) if isinstance(wd, list) else wd
            if day.weekday() >= 6 or rng.random() > days / 6:
                return 0.0
            if protested or work == "take_day_off":
                return 0.0
            if "net_per_month" in inc:
                base = in_range(value(inc["net_per_month"]), pos) / 26
            else:
                base = in_range(value(inc.get("gross_per_working_day") or
                                      inc.get("net_per_working_day")), pos)
            v = value(inc.get("variability", 0.0))
            amount = base * rng.uniform(1 - v, 1 + v)
            return amount * (EXTRA_WORK_INCOME if work == "work_extra" else 1)
        if pattern == "monthly":
            if day.day != value(self.cal["salary_day"]):
                return 0.0
            gross = in_range(value(inc["gross_per_month"]), pos)
            ded = SALARY_DEDUCTIONS if inc.get("deductions") else 0.0
            return gross * (1 - ded)
        if pattern == "seasonal":
            amount = in_range(value(inc["small_regular_sales_per_week"]),
                              pos) / 7
            harvest = value(self.cal["long_rains_harvest"])[0]
            if day.day == 1 and f"{day.month:02d}" == harvest:
                amount += in_range(value(inc["harvest_income_per_season"]),
                                   pos)
            if "cash transfer" in self.world.personas[aid]["livelihood"] \
                    and day.day == 15:
                amount += value(inc["cash_transfer_per_month"])
            return amount
        if pattern == "irregular":
            if day.weekday() != 0:
                return 0.0
            v = value(inc.get("variability", 0.5))
            return in_range(value(inc["per_month"]), pos) / 4 * \
                rng.uniform(1 - v, 1 + v)
        return 0.0

    # ---- obligations ----

    def _due(self, ob: dict, day: date) -> bool:
        f = ob["frequency"]
        if f == "daily":
            return True
        if f == "working_day":
            return day.weekday() < 6
        if f == "weekly":
            return day.weekday() == 0
        if f == "monthly":
            due_day = value(self.cal["rent_due_day"]) if ob.get("due") \
                == "rent_due_day" else 1
            return day.day == due_day
        if f == "term":
            return any(day == date.fromisoformat(s)
                       for s, _ in value(self.cal["school_terms_2024"]))
        if f == "season":
            return (day.month, day.day) in ((3, 15), (10, 15))
        if f == "yearly":
            return (day.month, day.day) == (1, 15)
        return False

    def _upcoming(self, st: dict, day: date, horizon: int = 14) -> list[str]:
        out = []
        for ob in st["obligations"]:
            if ob["frequency"] in ("daily", "working_day", "weekly"):
                continue
            for k in range(1, horizon + 1):
                d = day + timedelta(days=k)
                if self._due(ob, d):
                    out.append(f"{ob['name'].replace('_', ' ')} KES "
                               f"{ob['kes']:,} due in {k} days")
                    break
        return out

    # ---- daily bookkeeping ----

    def _book_day(self, aid: int, day: date) -> dict:
        st = self.state[aid]
        protested = bool(self.civic and self.civic.protested(aid, day))
        income = self._income(aid, st, day, protested)
        st["cash"] += income
        spent: dict[str, float] = {}
        missed: list[str] = []
        plan = SPENDING_PLANS[st["spending_plan"]]
        working = income > 0 or self.world.profile(aid)["income"]["pattern"] \
            != "daily"
        for ob in st["obligations"]:
            if not self._due(ob, day):
                continue
            amount = float(ob["kes"])
            if ob["name"] in ("fuel", "transport") and not working:
                continue  # no work, no fuel or fare
            if ob["name"] == "food" or ob["name"] == "food_purchases":
                amount *= plan * self._food_factor(day)
                amount *= 1 + OTHER_GOODS_SHARE_OF_FOOD * (
                    self._multiplier("household_goods", day) - 1)
                paid = min(amount, max(st["cash"], 0.0))
                if paid < amount * 0.999:
                    missed.append(f"{ob['name']} (short by KES "
                                  f"{amount - paid:,.0f})")
                st["cash"] -= paid
                spent[ob["name"]] = spent.get(ob["name"], 0) + paid
                continue
            if st["cash"] >= amount:
                st["cash"] -= amount
                spent[ob["name"]] = spent.get(ob["name"], 0) + amount
            else:
                st["arrears"][ob["name"]] = st["arrears"].get(
                    ob["name"], 0) + amount
                missed.append(ob["name"])
        total = sum(spent.values())
        fees = total * MOBILE_MONEY_FEE_SHARE * self._multiplier(
            "mobile_money_fees", day)
        st["cash"] -= min(fees, max(st["cash"], 0.0))
        spent["mobile_money_fees"] = round(fees, 1)
        # Loans accrue interest daily.
        st["debt"] *= 1 + st.get("debt_rate", 0.0) / 30
        row = {"date": day.isoformat(), "agent_id": aid,
               "profile_id": st["profile_id"], "income": round(income, 1),
               "spent": {k: round(v, 1) for k, v in spent.items()},
               "spent_total": round(sum(spent.values()), 1),
               "missed": missed, "cash_end": round(st["cash"], 1),
               "debt_end": round(st["debt"], 1),
               "arrears_end": {k: round(v) for k, v in st["arrears"].items()},
               "spending_plan": st["spending_plan"],
               "work_plan": st["work_plan"], "protested": protested,
               "borrowed": st["loans_today"]}
        st["last_day"] = row
        st["loans_today"] = []
        if st["work_plan"] == "take_day_off":
            st["work_plan"] = "normal"  # a day off lasts one day
        return row

    # ---- observation (called by code, not by the model) ----

    def observe_for(self, agent_id: int) -> str:
        st = self.state[agent_id]
        day = self.t.date()
        lines = [f"Your money today ({day.strftime('%A %d %B %Y')}): "
                 f"cash KES {st['cash']:,.0f}."]
        last = st["last_day"]
        if last:
            parts = ", ".join(f"{k.replace('_', ' ')} {v:,.0f}"
                              for k, v in last["spent"].items() if v)
            lines.append(f"Last time you earned KES {last['income']:,.0f} "
                         f"and spent KES {last['spent_total']:,.0f}"
                         f"{' (' + parts + ')' if parts else ''}.")
            if last["missed"]:
                lines.append("You could not pay: " + ", ".join(
                    m.replace("_", " ") for m in last["missed"]) + ".")
        if st["arrears"]:
            lines.append("You owe (unpaid bills): " + ", ".join(
                f"{k.replace('_', ' ')} KES {v:,.0f}"
                for k, v in st["arrears"].items()) + ".")
        if st["debt"] > 0:
            lines.append(f"Loans outstanding: KES {st['debt']:,.0f}.")
        upcoming = self._upcoming(st, day)
        if upcoming:
            lines.append("Coming up: " + "; ".join(upcoming) + ".")
        lines.append(f"Spending plan: {st['spending_plan']}; "
                     f"work plan today: {st['work_plan']}.")
        return "\n".join(lines)

    # ---- tools for the model ----

    @tool(readonly=False)
    async def set_spending_plan(self, agent_id: int, plan: str) -> str:
        """Change how your household spends on food and daily needs.

        :param agent_id: Your own agent id.
        :param plan: "normal", "cut_back" (eat less, buy less) or "cheaper_goods" (switch to cheaper brands and items).
        :returns: Confirmation.
        """
        check_agent(agent_id)
        if plan not in SPENDING_PLANS:
            raise ValueError(f"plan must be one of {list(SPENDING_PLANS)}")
        self.state[agent_id]["spending_plan"] = plan
        self._log_action(agent_id, "set_spending_plan", plan=plan)
        return f"Spending plan is now {plan}."

    @tool(readonly=False)
    async def set_work_plan(self, agent_id: int, plan: str) -> str:
        """Decide how to work today.

        :param agent_id: Your own agent id.
        :param plan: "normal", "work_extra" (longer hours, more income, more fuel) or "take_day_off" (no income today).
        :returns: Confirmation.
        """
        check_agent(agent_id)
        if plan not in WORK_PLANS:
            raise ValueError(f"plan must be one of {list(WORK_PLANS)}")
        self.state[agent_id]["work_plan"] = plan
        self._log_action(agent_id, "set_work_plan", plan=plan)
        return f"Work plan is now {plan}."

    @tool(readonly=False)
    async def borrow(self, agent_id: int, source: str, amount: float) -> str:
        """Borrow money.

        :param agent_id: Your own agent id.
        :param source: "mobile_loan" (Fuliza, M-Shwari), "chama", "family" or "shylock".
        :param amount: Amount in KES.
        :returns: Whether the loan was given.
        """
        check_agent(agent_id)
        if source not in LOAN_TERMS:
            raise ValueError(f"source must be one of {list(LOAN_TERMS)}")
        rate, months = LOAN_TERMS[source]
        st = self.state[agent_id]
        limit = 30 * self._daily_cost_estimate(st) * months
        if amount <= 0 or st["debt"] + amount > limit + 1:
            raise ValueError(f"loan refused; you can borrow at most KES "
                             f"{max(limit - st['debt'], 0):,.0f} more")
        st["cash"] += amount
        st["debt"] += amount
        st["debt_rate"] = max(st.get("debt_rate", 0.0), rate)
        st["loans_today"].append({"source": source, "amount": amount})
        self._log_action(agent_id, "borrow", source=source, amount=amount)
        return f"You received KES {amount:,.0f} from {source}."

    @tool(readonly=False)
    async def pay_bill(self, agent_id: int, bill: str, amount: float) -> str:
        """Pay off an unpaid bill or a loan.

        :param agent_id: Your own agent id.
        :param bill: The bill name as shown under "You owe", or "loan".
        :param amount: Amount in KES.
        :returns: What is still owed.
        """
        check_agent(agent_id)
        st = self.state[agent_id]
        key = bill.strip().lower().replace(" ", "_")
        owed = st["debt"] if key == "loan" else st["arrears"].get(key)
        if owed is None:
            raise ValueError(f"no unpaid bill called {bill!r}")
        amount = min(amount, owed, st["cash"])
        if amount <= 0:
            raise ValueError("not enough cash")
        st["cash"] -= amount
        if key == "loan":
            st["debt"] -= amount
        else:
            st["arrears"][key] -= amount
            if st["arrears"][key] < 1:
                del st["arrears"][key]
        self._log_action(agent_id, "pay_bill", bill=key, amount=amount)
        return f"Paid KES {amount:,.0f}; KES {owed - amount:,.0f} still owed."

    def _log_action(self, agent_id: int, action: str, **extra) -> None:
        append_jsonl(self.log_path, {
            "type": "action", "date": self.t.date().isoformat(),
            "agent_id": agent_id, "action": action, **extra})

    async def step(self, tick: int, t: datetime):
        """Book every day from t to t + tick for every household."""
        days = max(1, round(tick / 86400))
        for k in range(days):
            day = t.date() + timedelta(days=k)
            for aid in self.world.personas:
                append_jsonl(self.log_path, {
                    "type": "household_day", **self._book_day(aid, day)})
