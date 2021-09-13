# BGNBD-GG_CLTV_Prediction

In this project, Customer Lifetime Value is predicted by using a probabilistic approach with BG/NBD-GG models. 

For more details, please see my medium article:

https://kubrasen84.medium.com/customer-lifetime-value-prediction-based-on-statistical-bg-nbd-gamma-gamma-models-e791186245b2

#Dataset
Online_Retail_II dataset contains the invoices of the transactions between 01/12/2009 - 09/12/2011 of a UK-based e-trade company

Variables:
InvoiceNo: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.
StockCode: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
Description: Product (item) name. Nominal.
Quantity: The quantities of each product (item) per transaction. Numeric.
InvoiceDate: Invice date and time. Numeric. The day and time when a transaction was generated.
UnitPrice: Unit price. Numeric. Product price per unit in sterling (Â£).
CustomerID: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
Country: Country name. Nominal. The name of the country where a customer resides.
