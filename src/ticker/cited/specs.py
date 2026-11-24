"""Drafted question specifications: the text half of the question set.

Each spec records a `locator` -- a phrase copied out of the filing -- rather
than the chunk ids the answer was found at. `ticker.cited.build.build_text`
re-runs the substring scan for that phrase and derives `gold_chunk_ids` from
it, so the gold set has a reproducible provenance chain back to the filing
text instead of being a list of ids someone pasted in.

`max_chunks` bounds how many chunks a locator may hit. Chunking is 4-sentence
windows at stride 2, so a phrase near a window boundary legitimately lands in
two or three adjacent chunks, and some filings repeat a risk sentence verbatim
across two risk factors. A gold set of "every chunk containing this claim" is
the right target for citation recall -- any of them supports the claim -- so
the bound exists to catch a locator that has broadened into something generic,
not to insist on a single chunk. Zero matches or more than `max_chunks` fails
the build.

Unanswerable specs carry `absence_probes` instead: phrases that must match
zero chunks across that company's indexed filings. That makes unanswerability
a checked property rather than an assertion, and the check has already earned
its place -- three earlier drafts of this set were wrong. A question asking
for NVIDIA's fiscal 2019 revenue is answerable, because the fiscal 2021 10-K
carries fiscal 2019 as a comparative. A question asking for Qualcomm's
effective tax rate is answerable, because MD&A discusses it. And an attempt to
build these around balance-sheet figures failed outright: the 486 indexed 8-K
EX-99.1 earnings releases contain full balance sheets and income statements,
so "total assets" is in this corpus even though Item 8 is not extracted.

What is structurally absent, and therefore safe ground for these questions:
cover-page facts, Item 5 content (dividend declarations, the issuer repurchase
table), proxy-statement content, note-level detail that only Item 8 carries,
fiscal years outside the 2020-2025 window, and events that did not happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Source:
    """One filing a gold answer draws on."""

    company: str
    fiscal_year: int
    item: str
    locator: str
    max_chunks: int = 4
    form_type: str = "10-K"


@dataclass(frozen=True, slots=True)
class TextSpec:
    question_id: str
    kind: str
    text: str
    gold_answer: str
    rationale: str
    sources: tuple[Source, ...] = ()
    absence_probes: tuple[str, ...] = ()
    absence_company: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


QUALITATIVE: tuple[TextSpec, ...] = (
    TextSpec(
        question_id="q01",
        kind="qualitative",
        text="What share of NVIDIA's fiscal 2025 revenue came from sales outside the United States, and what does the company say limits its offerings in the China market?",
        gold_answer="NVIDIA states it generated 53% of its fiscal 2025 revenue from sales outside the United States, and that its offerings in the China market are limited by export controls, in a market it describes as highly competitive and expected to remain so.",
        rationale="Both facts are stated in the same sentence pair of the international-operations risk factor.",
        sources=(Source("NVDA", 2025, "1A", "we generated 53% of our revenue in fiscal year 2025 from sales outside the United States"),),
        tags=("risk_factors", "geographic_concentration"),
    ),
    TextSpec(
        question_id="q02",
        kind="qualitative",
        text="What does Qualcomm say about Apple's future use of its modem products, and what effect does it expect?",
        gold_answer="Qualcomm says it expects Apple will increasingly use its own modem products rather than Qualcomm's in future devices, which it expects to have a significant negative impact on QCT revenues, results of operations and cash flows.",
        rationale="Stated in the customer-concentration risk factor. Qualcomm repeats the sentence in a second risk factor, so the locator legitimately matches four chunks.",
        sources=(Source("QCOM", 2025, "1A", "we expect that Apple will increasingly use its own modem products"),),
        tags=("risk_factors", "customer_concentration"),
    ),
    TextSpec(
        question_id="q03",
        kind="qualitative",
        text="What does Zions say drove the improvement in its accumulated other comprehensive income during 2024?",
        gold_answer="Zions says AOCI improved $312 million compared with December 31, 2023, primarily due to $194 million of unrealized loss amortization on securities previously transferred from available-for-sale to held-to-maturity and $31 million related mainly to paydowns on AFS securities, with a further $87 million decrease in unrealized losses on risk-management derivatives.",
        rationale="The drivers and their amounts are itemised in the capital discussion of MD&A.",
        sources=(Source("ZION", 2024, "7", "AOCI improved $312 million"),),
        tags=("highlighted_change", "mdna"),
    ),
    TextSpec(
        question_id="q04",
        kind="qualitative",
        text="What does KeyCorp's Item 3 Legal Proceedings disclose, and how does it say it assesses legal reserves?",
        gold_answer="Item 3 does not describe individual proceedings; it incorporates the Legal Proceedings section of Note 22, Commitments, Contingent Liabilities, and Guarantees, by reference. KeyCorp says it assesses its liabilities and contingencies at least quarterly, records a liability where a loss is probable and reasonably estimable, and may increase or decrease those reserves quarterly as developments warrant.",
        rationale="Item 3 is an incorporation by reference plus a statement of reserve policy. The referenced note sits in Item 8, which this corpus does not extract.",
        sources=(Source("KEY", 2024, "3", "is incorporated herein by reference"),),
        tags=("legal_proceedings", "incorporation_by_reference"),
    ),
    TextSpec(
        question_id="q05",
        kind="qualitative",
        text="What does Intel say about becoming a foundry for external customers, and how does it describe its manufacturing capacity strategy?",
        gold_answer="Intel says it is seeking to expand as a third-party foundry for external customers, and that it is positioning itself to create a resilient global semiconductor supply chain by investing in geographically balanced manufacturing capacity.",
        rationale="Both statements appear in the strategy discussion of Item 1.",
        sources=(Source("INTC", 2024, "1", "seeking to expand as a third-party foundry for external customers"),),
        tags=("strategy",),
    ),
    TextSpec(
        question_id="q06",
        kind="qualitative",
        text="Which wafer fabrication facility did Microchip announce it would close, when did the closure complete, and what annual savings does it expect?",
        gold_answer="Microchip announced on December 2, 2024 that it would close its Tempe, Arizona wafer fabrication facility, referred to as Fab 2. The closure was completed in May 2025 and Microchip expects it to generate annual cash savings of approximately $90 million.",
        rationale="All three facts are stated in the Business and Macroeconomic Environment discussion of MD&A.",
        sources=(Source("MCHP", 2025, "7", "close our Tempe, Arizona wafer fabrication facility"),),
        tags=("highlighted_change", "mdna"),
    ),
    TextSpec(
        question_id="q07",
        kind="qualitative",
        text="How does Broadcom describe the two kinds of solutions it supplies, and which acquisitions does it name as part of its history?",
        gold_answer="Broadcom describes itself as supplying a broad range of semiconductor and semiconductor-based solutions and infrastructure software solutions. It names LSI Corporation, Broadcom Corporation, Brocade Communications Systems, CA, Symantec Enterprise Security and VMware among the acquisitions through which it evolved, with origins in AT&T/Bell Labs, Lucent and Hewlett-Packard.",
        rationale="Stated in the opening overview of Item 1. The locator names one acquisition rather than the generic phrase 'infrastructure software solutions', which recurs throughout Item 1.",
        sources=(Source("AVGO", 2025, "1", "Symantec Enterprise Security"),),
        tags=("segments", "strategy"),
    ),
    TextSpec(
        question_id="q08",
        kind="qualitative",
        text="What share of Microchip's fiscal 2025 net sales came through distributors, and which distributor accounted for 10% or more of net sales?",
        gold_answer="Microchip derived 45% of fiscal 2025 net sales through distributors and 55% from customers serviced directly, compared with 47% and 53% in fiscal 2024. Arrow Electronics, its largest distributor, accounted for 10% of net sales in fiscal 2025 and 12% in fiscal 2024; no other distributor or direct customer exceeded 10%.",
        rationale="All figures are stated in one passage of the sales-channel discussion in Item 1.",
        sources=(Source("MCHP", 2025, "1", "we derived 45% of our net sales through distributors"),),
        tags=("customer_concentration",),
    ),
    TextSpec(
        question_id="q09",
        kind="qualitative",
        text="How much capital did Texas Instruments say it allocated over the ten years from 2015 to 2024, and how much of that went to capital expenditures?",
        gold_answer="Texas Instruments says it allocated $101 billion over the ten-year period from 2015 to 2024, of which about $20 billion went to capital expenditures. It says the largest allocation over the period was to drive organic growth, which includes R&D, sales and marketing, capital expenditures and working capital for inventory.",
        rationale="Stated in the capital-allocation discussion of Item 1.",
        sources=(Source("TXN", 2024, "1", "we allocated $101 billion"),),
        tags=("strategy", "capital_allocation"),
    ),
    TextSpec(
        question_id="q10",
        kind="qualitative",
        text="What does Qualcomm's Item 3 Legal Proceedings say?",
        gold_answer="Qualcomm's Item 3 describes no proceedings itself; it states that information regarding certain legal proceedings is provided in Note 7, Commitments and Contingencies, of the Notes to Consolidated Financial Statements.",
        rationale="Item 3 is a single-sentence cross-reference. The referenced note is in Item 8, which this corpus does not extract.",
        sources=(Source("QCOM", 2025, "3", "Note 7. Commitments and Contingencies"),),
        tags=("legal_proceedings", "incorporation_by_reference"),
    ),
    TextSpec(
        question_id="q11",
        kind="qualitative",
        text="Which captions and note does NXP's Item 3 Legal Proceedings incorporate by reference?",
        gold_answer="NXP's Item 3 incorporates by reference the information under the Legal Proceedings and Environmental Remediation captions of Note 15 to the Consolidated Financial Statements, which is included in Part II, Item 8, and directs readers to Part I, Item 1A Risk Factors for related risk discussion.",
        rationale="Stated in the whole of Item 3, which is two sentences long.",
        sources=(Source("NXPI", 2024, "3", "Environmental Remediation"),),
        tags=("legal_proceedings", "incorporation_by_reference"),
    ),
    TextSpec(
        question_id="q12",
        kind="qualitative",
        text="Which note does Regions Financial's Item 3 Legal Proceedings incorporate by reference?",
        gold_answer="Regions' Item 3 incorporates by reference the Legal Contingencies section of Note 23, Commitments, Contingencies and Guarantees, of the Notes to Consolidated Financial Statements included in Item 8, and also points to Note 12 on regulatory capital requirements and restrictions.",
        rationale="Stated in Item 3, which is a cross-reference only.",
        sources=(Source("RF", 2024, "3", "Legal Contingencies section of Note 23"),),
        tags=("legal_proceedings", "incorporation_by_reference"),
    ),
    TextSpec(
        question_id="q13",
        kind="qualitative",
        text="Why does ON Semiconductor say government incentives to its competitors could put it at a disadvantage, and which end-markets does it say most of its revenue depends on?",
        gold_answer="ON Semiconductor says some competitors may receive governmental subsidies or incentives that give them a competitive advantage, citing US and European Union legislation funding semiconductor research, development and manufacturing; if ON cannot access such funding, or competitors receive more, it may be at a disadvantage in developing and producing products, affecting market share, revenue and profitability. It says a significant portion of its revenue comes from customers in the automotive and industrial end-markets.",
        rationale="Both points appear in adjacent risk factors in the same passage.",
        sources=(Source("ON", 2024, "1A", "some of our competitors may receive governmental subsidies"),),
        tags=("risk_factors",),
    ),
    TextSpec(
        question_id="q14",
        kind="qualitative",
        text="How does Texas Instruments describe the cyclicality of the semiconductor market in its risk factors, and what customer behaviour does it say may hurt results?",
        gold_answer="Texas Instruments says the cyclical nature of the semiconductor market occasionally leads to significant and rapid increases and decreases in product demand. It says the loss or significant curtailment of purchases by one or more large customers may adversely affect results, including curtailments from a change in those customers' design or manufacturing sourcing policies, the timing of customer or distributor inventory adjustments, changes in demand for customer products, or trade restrictions.",
        rationale="Stated in the demand risk factor of Item 1A.",
        sources=(Source("TXN", 2024, "1A", "cyclical nature of the semiconductor market"),),
        tags=("risk_factors",),
    ),
    TextSpec(
        question_id="q15",
        kind="qualitative",
        text="What does Micron say changed in memory market conditions during 2024 compared with 2023, and what cost measures did it reverse?",
        gold_answer="Micron says that throughout 2024 it saw substantial improvements in pricing and margins versus 2023, as increasing demand growth driven partly by AI deployment, mostly normal customer inventories and industry-wide supply discipline brought supply and demand substantially better into balance than the 2023 downturn. In connection with those improved conditions it reinstated bonuses and phased out certain other temporary cost-saving measures implemented in 2023.",
        rationale="Stated in the overview discussion of MD&A.",
        sources=(Source("MU", 2025, "7", "we reinstated our bonuses"),),
        tags=("highlighted_change", "mdna"),
    ),
    TextSpec(
        question_id="q16",
        kind="qualitative",
        text="What conditions does Micron say government incentives typically require it to meet, and what risk does it identify?",
        gold_answer="Micron says government incentives, which may take the form of grants, subsidies, loans and tax arrangements, typically require it to achieve or maintain certain levels of investment, capital spending, employment, technology deployment or development milestones, construction or production milestones, or research and development activities, and could restrict it from undertaking certain activities. It says it may fail to achieve those milestones on time or at all for reasons partly outside its control, including a cyclical or global downturn.",
        rationale="Stated in the government-incentives risk factor.",
        sources=(Source("MU", 2025, "1A", "grants, subsidies, loans, and tax arrangements"),),
        tags=("risk_factors",),
    ),
    TextSpec(
        question_id="q17",
        kind="qualitative",
        text="How does Analog Devices describe the Intelligent Edge, and what does it say is central to its strategy?",
        gold_answer="Analog Devices describes the Intelligent Edge as characterised by ubiquitous sensing, hyper-scale and edge computing, artificial intelligence and pervasive connectivity, trends that drive applications expanding demand for its high-performance analog, mixed-signal, power and RF integrated circuits. It says central to its strategy is a focus on solving challenges its customers face across the most impactful application areas.",
        rationale="Stated in the strategy discussion of Item 1.",
        sources=(Source("ADI", 2025, "1", "The Intelligent Edge is characterized by ubiquitous sensing"),),
        tags=("strategy",),
    ),
    TextSpec(
        question_id="q18",
        kind="qualitative",
        text="What does Cullen/Frost give as its operating objectives, and what is its stated stance on acquisitions?",
        gold_answer="Cullen/Frost says its operating objectives include expansion, diversification within its markets, growth of fee-based income, and growth internally and, to a lesser extent, through acquisitions of financial institutions, branches and financial services businesses. It says it is currently focused on organic growth but may seek culturally similar merger or acquisition partners with experienced management and either significant market presence or potential for improved profitability, and that it evaluates opportunities from time to time. It also states it is not dependent on any single industry or customer.",
        rationale="Stated in the business description of Item 1.",
        sources=(Source("CFR", 2024, "1", "Our operating objectives include expansion"),),
        tags=("strategy",),
    ),
    TextSpec(
        question_id="q19",
        kind="qualitative",
        text="What does Western Alliance say the bank closures of early 2023 did to mid-sized banks including itself?",
        gold_answer="Western Alliance says the bank closures in the first half of 2023 led to disruption and volatility, including deposit outflows, at many mid-sized banks including itself, increasing its need for liquidity. It frames this within a risk factor about events affecting financial institutions, or rumours of them, eroding customer confidence in the banking system and causing deposit volatility, liquidity issues, credit problems and losses.",
        rationale="Stated in the financial-institution-stress risk factor of the fiscal 2023 filing.",
        sources=(Source("WAL", 2023, "1A", "bank closures in the first half of 2023"),),
        tags=("risk_factors", "highlighted_change"),
    ),
    TextSpec(
        question_id="q20",
        kind="qualitative",
        text="How does NVIDIA define its direct customers in the revenue concentration discussion?",
        gold_answer="NVIDIA says it refers to customers who purchase products directly from NVIDIA as direct customers, and gives as examples add-in board manufacturers, distributors, original design manufacturers, original equipment manufacturers and system integrators. It notes some customers may buy directly and use either internal resources or third-party system integrators to complete their build.",
        rationale="Stated in the Concentration of Revenue discussion of MD&A.",
        sources=(Source("NVDA", 2025, "7", "We refer to customers who purchase products directly from NVIDIA as direct customers"),),
        tags=("customer_concentration", "mdna"),
    ),
    TextSpec(
        question_id="q21",
        kind="qualitative",
        text="Why did Zions' preferred stock balance fall during 2024, and what effect did that have on earnings available to common shareholders?",
        gold_answer="Zions says preferred stock decreased $374 million because it redeemed the outstanding shares of its Series G, I and J preferred stock in the fourth quarter of 2024. The redemption caused a one-time reduction of approximately $6 million to net earnings applicable to common shareholders, from recognising capitalised preferred stock issuance costs.",
        rationale="Stated in the capital discussion of MD&A, and repeated in the MD&A overview, so the locator matches four chunks.",
        sources=(Source("ZION", 2024, "7", "Preferred stock decreased $374 million"),),
        tags=("highlighted_change", "mdna"),
    ),
    TextSpec(
        question_id="q22",
        kind="qualitative",
        text="What does Qualcomm say the concentration of device share among a few companies may do to its prices?",
        gold_answer="Qualcomm says the concentration of device share among a few companies, and those companies' corresponding purchasing power, may result in lower prices for its products, which could adversely affect its revenues and margins.",
        rationale="Stated in the customer-concentration risk factor, a distinct claim from the Apple-specific passage behind q02.",
        sources=(Source("QCOM", 2025, "1A", "the concentration of device share among a few companies"),),
        tags=("risk_factors", "customer_concentration"),
    ),
    TextSpec(
        question_id="q23",
        kind="qualitative",
        text="Which Intel Core Ultra product does Intel say was manufactured by an external foundry, and which process made the majority of its 2024 client sales?",
        gold_answer="Intel says the Intel Core Ultra 200V Series was manufactured by an external foundry and leverages its Xe2 GPU architecture. It says a significant majority of its 2024 client sales consisted of 13th and 14th Gen Intel Core processors, manufactured on Intel 7.",
        rationale="Both statements appear in the client computing product discussion of Item 1.",
        sources=(Source("INTC", 2024, "1", "The 200V Series, manufactured by an external foundry"),),
        tags=("segments", "strategy"),
    ),
    TextSpec(
        question_id="q24",
        kind="qualitative",
        text="What are Huntington Bancshares' two business segments, and what does the Consumer and Regional Banking segment provide?",
        gold_answer="Huntington reports two business segments, Consumer & Regional Banking and Commercial Banking, alongside a Treasury / Other function. It says the Consumer & Regional Banking segment provides a wide array of financial products and services to consumer and business customers, including deposits, lending, payments, mortgage banking, dealer financing, investment management, trust, brokerage and insurance.",
        rationale="Stated in the segment description of Item 1.",
        sources=(Source("HBAN", 2024, "1", "Consumer & Regional Banking and Commercial Banking"),),
        tags=("segments",),
    ),
    TextSpec(
        question_id="q25",
        kind="qualitative",
        text="How many active banking subsidiaries did Comerica own at the end of 2024, and what were they?",
        gold_answer="Comerica says that as of December 31, 2024 it owned, directly or indirectly, all the outstanding common stock of two active banking subsidiaries: Comerica Bank, a Texas banking association, and Comerica Bank & Trust, National Association, as well as non-banking subsidiaries. It also says it was among the 25 largest commercial US financial holding companies by total assets.",
        rationale="Stated in the corporate description of Item 1.",
        sources=(Source("CMA", 2024, "1", "two active banking subsidiaries"),),
        tags=("segments",),
    ),
)


MULTI_FILING: tuple[TextSpec, ...] = (
    TextSpec(
        question_id="m01",
        kind="multi_filing",
        text="How did Western Alliance's risk-factor language about stress at other financial institutions change between its fiscal 2022 and fiscal 2023 10-K?",
        gold_answer="The fiscal 2023 10-K adds a concrete account of the early-2023 bank failures that the fiscal 2022 10-K does not contain: it says the bank closures in the first half of 2023 led to disruption and volatility, including deposit outflows, at many mid-sized banks including Western Alliance, increasing its need for liquidity. The fiscal 2022 filing frames adverse conditions generically, as a worsening of business and economic conditions that could cause a decrease in deposit balances or in demand for loans and other products, with no reference to any actual closures.",
        rationale="The phrase 'bank closures in the first half of 2023' appears in the fiscal 2023 filing and in no fiscal 2022 filing, so answering requires reading both years.",
        sources=(
            Source("WAL", 2023, "1A", "bank closures in the first half of 2023"),
            Source("WAL", 2022, "1A", "a decrease in deposit balances or the demand for loans"),
        ),
        tags=("risk_factors", "wording_change"),
    ),
    TextSpec(
        question_id="m02",
        kind="multi_filing",
        text="How did NVIDIA's international-operations risk factor change between the fiscal 2023 and fiscal 2025 10-K, in both the share of revenue earned outside the United States and what it says about China?",
        gold_answer="The share fell: the fiscal 2023 filing says 69% of fiscal 2023 revenue came from sales outside the United States, while the fiscal 2025 filing says 53% of fiscal 2025 revenue did. The fiscal 2025 filing also adds that NVIDIA's offerings in China are limited by export controls and that the market is highly competitive and expected to remain so, a statement the fiscal 2023 risk factor does not make.",
        rationale="Each percentage is stated only in its own filing, and the export-control sentence is new in the later one, so neither filing answers the question alone.",
        sources=(
            Source("NVDA", 2025, "1A", "we generated 53% of our revenue in fiscal year 2025 from sales outside the United States"),
            Source("NVDA", 2023, "1A", "we generated 69% of our revenue during fiscal year 2023"),
        ),
        tags=("risk_factors", "wording_change"),
    ),
    TextSpec(
        question_id="m03",
        kind="multi_filing",
        text="How did Microchip's account of its business and macroeconomic environment change between the fiscal 2024 and fiscal 2025 10-K?",
        gold_answer="The fiscal 2025 10-K describes navigating a large inventory correction, with the overall macroeconomic environment weak throughout the year but improvement signs from the March 2024 quarter such as fewer push-out and cancellation requests and more expedites and pull-ins, and it announces the closure of the Tempe Fab 2 facility. The fiscal 2024 filing describes the onset instead, with customers feeling slowing economic activity, business uncertainty, persistent inflation and higher interest rates, and requesting push-outs or cancellations of backlog to reduce inventory.",
        rationale="The phrase 'inventory correction' appears in the fiscal 2025 filing and not the fiscal 2024 one.",
        sources=(
            Source("MCHP", 2025, "7", "inventory correction"),
            Source("MCHP", 2024, "7", "push out or cancel backlog"),
        ),
        tags=("mdna", "wording_change"),
    ),
    TextSpec(
        question_id="m04",
        kind="multi_filing",
        text="How did the share of Microchip's net sales coming through distributors change between its fiscal 2023 and fiscal 2025 10-K?",
        gold_answer="The distributor share fell. The fiscal 2025 10-K says 45% of net sales came through distributors in fiscal 2025 and 47% in fiscal 2024, against 55% and 53% respectively from customers serviced directly. The fiscal 2023 10-K reports the share for fiscal 2023 and fiscal 2022, which were higher.",
        rationale="Each filing states only its own two years, so the span from fiscal 2023 to fiscal 2025 needs both.",
        sources=(
            Source("MCHP", 2025, "1", "of our net sales through distributors"),
            Source("MCHP", 2023, "1", "of our net sales through distributors"),
        ),
        tags=("customer_concentration", "wording_change"),
    ),
    TextSpec(
        question_id="m05",
        kind="multi_filing",
        text="How did the ten-year capital allocation figure Texas Instruments reports change between its fiscal 2021 and fiscal 2024 10-K, and which period does each cover?",
        gold_answer="Each filing reports a rolling ten-year window, so the period moves and the total changes. The fiscal 2024 10-K reports $101 billion allocated over the ten years from 2015 to 2024, of which about $20 billion went to capital expenditures. The fiscal 2021 10-K reports the ten years from 2012 to 2021 instead, with a different total.",
        rationale="Both filings use the same sentence structure with a different window and total, so comparing them requires both.",
        sources=(
            Source("TXN", 2024, "1", "we allocated $101 billion"),
            Source("TXN", 2021, "1", "Over a 10-year period from 2012 to 2021"),
        ),
        tags=("strategy", "wording_change"),
    ),
)


UNANSWERABLE: tuple[TextSpec, ...] = (
    TextSpec(
        question_id="u01",
        kind="unanswerable",
        text="How many shares of Intel common stock were outstanding as of the cover date of its fiscal 2024 10-K?",
        gold_answer="The indexed filings do not answer this. The outstanding share count appears on the 10-K cover page, which is not one of the items extracted into this corpus.",
        rationale="Cover-page facts are not extracted. dei:EntityCommonStockSharesOutstanding is tagged for only one of the twenty companies and findable in the indexed text of none.",
        absence_company="INTC",
        absence_probes=(
            "shares of common stock outstanding as of",
            "shares outstanding as of the close of business",
        ),
        tags=("not_indexed", "cover_page"),
    ),
    TextSpec(
        question_id="u02",
        kind="unanswerable",
        text="What was Micron's revenue in fiscal 2014?",
        gold_answer="The indexed filings do not answer this. The corpus covers filings made from 2020 onward, and fiscal 2014 is far enough back that it appears in none of them, not even as a comparative.",
        rationale="No indexed Micron filing mentions fiscal 2014 at all.",
        absence_company="MU",
        absence_probes=("fiscal 2014", "fiscal year 2014"),
        tags=("year_not_indexed",),
    ),
    TextSpec(
        question_id="u03",
        kind="unanswerable",
        text="What was NVIDIA's revenue in fiscal 2015?",
        gold_answer="The indexed filings do not answer this. The corpus covers filings made from 2020 onward and no indexed NVIDIA filing reports or mentions fiscal 2015.",
        rationale="No indexed NVIDIA filing mentions fiscal 2015. Note that fiscal 2019 would be answerable, because the fiscal 2021 10-K carries it as a comparative; the window has to be pushed back further than the corpus start date to get a genuine miss.",
        absence_company="NVDA",
        absence_probes=("fiscal 2015", "fiscal year 2015"),
        tags=("year_not_indexed",),
    ),
    TextSpec(
        question_id="u04",
        kind="unanswerable",
        text="What was Broadcom's revenue in fiscal 2016?",
        gold_answer="The indexed filings do not answer this. The corpus covers filings made from 2020 onward and no indexed Broadcom filing reports or mentions fiscal 2016.",
        rationale="No indexed Broadcom filing mentions fiscal 2016.",
        absence_company="AVGO",
        absence_probes=("fiscal 2016", "fiscal year 2016"),
        tags=("year_not_indexed",),
    ),
    TextSpec(
        question_id="u05",
        kind="unanswerable",
        text="What did Zions say about its acquisition of Comerica?",
        gold_answer="The indexed filings do not answer this, because no such acquisition is described in them. Zions and Comerica are both separately indexed registrants in this corpus and neither filing reports a combination between them.",
        rationale="An event that did not happen. No indexed Zions filing mentions Comerica at all.",
        absence_company="ZION",
        absence_probes=("Comerica",),
        tags=("event_did_not_happen",),
    ),
    TextSpec(
        question_id="u06",
        kind="unanswerable",
        text="What did Texas Instruments disclose about its acquisition of Analog Devices?",
        gold_answer="The indexed filings do not answer this, because no such acquisition is described in them. Both companies are separately indexed registrants here and no Texas Instruments filing reports a transaction with Analog Devices.",
        rationale="An event that did not happen. No indexed Texas Instruments filing mentions Analog Devices at all.",
        absence_company="TXN",
        absence_probes=("Analog Devices",),
        tags=("event_did_not_happen",),
    ),
    TextSpec(
        question_id="u07",
        kind="unanswerable",
        text="What did Huntington Bancshares disclose about acquiring KeyCorp?",
        gold_answer="The indexed filings do not answer this, because no such acquisition is described in them. No Huntington filing in this corpus mentions KeyCorp.",
        rationale="An event that did not happen. No indexed Huntington filing mentions KeyCorp at all.",
        absence_company="HBAN",
        absence_probes=("KeyCorp",),
        tags=("event_did_not_happen",),
    ),
    TextSpec(
        question_id="u08",
        kind="unanswerable",
        text="What was the balance of KeyCorp's recorded legal reserves at December 31, 2024?",
        gold_answer="The indexed filings do not answer this. KeyCorp's Item 3 states its reserve policy and incorporates Note 22 by reference, but that note sits in Item 8, which is not extracted into this corpus, so no reserve balance is retrievable.",
        rationale="Item 3 is a cross-reference and the referenced note is not extracted. Pairs with q04, which asks only what Item 3 itself discloses.",
        absence_company="KEY",
        absence_probes=("legal reserves of $", "aggregate range of reasonably possible losses"),
        tags=("not_indexed", "item_8"),
    ),
    TextSpec(
        question_id="u09",
        kind="unanswerable",
        text="What was the total compensation of Fifth Third's chief executive officer for 2024?",
        gold_answer="The indexed filings do not answer this. Executive compensation is disclosed in the annual proxy statement, a different filing from the 10-K, and the proxy is not part of this corpus.",
        rationale="Item 11 incorporates compensation disclosure from the proxy by reference, and the corpus holds only 10-K, 10-Q and 8-K exhibit text.",
        absence_company="FITB",
        absence_probes=("Summary Compensation Table", "total compensation of our chief executive"),
        tags=("not_in_corpus", "proxy"),
    ),
    TextSpec(
        question_id="u10",
        kind="unanswerable",
        text="How many shares did Analog Devices repurchase during the fourth quarter of fiscal 2025, and at what average price?",
        gold_answer="The indexed filings do not answer this. Quarterly issuer repurchase detail is disclosed in Item 5, which is not extracted into this corpus.",
        rationale="Item 5 is not among the extracted 10-K items, and the repurchase table lives there.",
        absence_company="ADI",
        absence_probes=("average price paid per share", "total number of shares purchased"),
        tags=("not_indexed", "item_5"),
    ),
)


ALL_TEXT_SPECS: tuple[TextSpec, ...] = QUALITATIVE + MULTI_FILING + UNANSWERABLE
