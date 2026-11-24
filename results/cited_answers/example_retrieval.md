Question `q08` (qualitative):

> What share of Microchip's fiscal 2025 net sales came through distributors, and which distributor accounted for 10% or more of net sales?

Gold answer, written from the filing before any system ran:

> Microchip derived 45% of fiscal 2025 net sales through distributors and 55% from customers serviced directly, compared with 47% and 53% in fiscal 2024. Arrow Electronics, its largest distributor, accounted for 10% of net sales in fiscal 2025 and 12% in fiscal 2024; no other distributor or direct customer exceeded 10%.

Gold passages, located by substring scan for `we derived 45% of our net sales through distributors`: `0000827054-25-000077#1#chunk100`, `0000827054-25-000077#1#chunk98`

What the `search` tool returns for this question (hybrid, k=3, filtered to MCHP 10-K). A tick marks a passage that is in the gold set.

1. `0000827054-25-000077#7#chunk94`
   MCHP 10-K item 7 (period ending 2025-03-31, filed 2025-05-22, accession 0000827054-25-000077)

   > Net sales of our other product line can fluctuate over time based on general economic and semiconductor industry conditions as well as changes in demand for our FPGA products, licenses, engineering services, memory products, timing systems, and manufacturing services (wafer foundry and assembly and test subcontracting). Distribution Distributors accounted for approximately 45% and 47% of our net sales in fiscal 2025 and fiscal 2024, respectively. With the exception of Arrow Electronics, our largest distributor, which accounted for 10% and 12% of our net sales in fiscal 2025 and in fiscal 2024, respectively, no other distributor or direct customer accounted for more than 10% of our net sales in fiscal 2025 or in fiscal 2024. Our distributors focus primarily on servicing the product requirements of a broad base of diverse customers. We believe that distributors provide an effective means of reaching this broad and diverse customer base and that customers recognize Microchip for its products and brand name and use distributors as an effective supply channel.

2. `0000827054-25-000077#7#chunk96`
   MCHP 10-K item 7 (period ending 2025-03-31, filed 2025-05-22, accession 0000827054-25-000077)

   > With the exception of Arrow Electronics, our largest distributor, which accounted for 10% and 12% of our net sales in fiscal 2025 and in fiscal 2024, respectively, no other distributor or direct customer accounted for more than 10% of our net sales in fiscal 2025 or in fiscal 2024. Our distributors focus primarily on servicing the product requirements of a broad base of diverse customers. We believe that distributors provide an effective means of reaching this broad and diverse customer base and that customers recognize Microchip for its products and brand name and use distributors as an effective supply channel. Generally, we do not have long-term agreements with our distributors and we, or our distributors, may terminate our relationships with each other with little or no advance notice. The loss of, or the disruption in the operations of, one or more of our distributors could reduce our future net sales in a given quarter and could result in an increase in inventory returns.

3. `0000827054-25-000077#1#chunk100` **gold**
   MCHP 10-K item 1 (period ending 2025-03-31, filed 2025-05-22, accession 0000827054-25-000077)

   > We believe that customers recognize us for our products and brand name and use distributors as an effective supply channel. In fiscal 2025, we derived 45% of our net sales through distributors compared to 55% of our net sales from customers serviced directly by us. In fiscal 2024, we derived 47% of our net sales through distributors compared to 53% of our net sales from customers serviced directly by us. With the exception of Arrow Electronics, our largest distributor, which made up 10% and 12% of our net sales, in fiscal 2025 and in fiscal 2024, respectively, no distributor or direct customer accounted for more than 10% of our net sales.

An answer over these passages would carry one claim per assertion, each citing the chunk ids above, and would be rejected if it cited an id the session never retrieved. No such answer exists yet: no evaluation run has been made.
