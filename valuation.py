def dcf_valuation(
    revenue,
    growth_rate,
    ebitda_margin,
    tax_rate,
    capex_pct,
    wc_pct,
    discount_rate,
    terminal_growth,
    debt,
    cash,
    shares
):
    years = 5
    fcfs = []

    for i in range(1, years + 1):
        rev = revenue * ((1 + growth_rate) ** i)
        ebitda = rev * ebitda_margin
        tax = ebitda * tax_rate
        nopat = ebitda - tax

        capex = rev * capex_pct
        wc = rev * wc_pct

        fcf = nopat - capex - wc
        fcfs.append(fcf)

    # Present Value of FCFs
    pv_fcfs = [
        fcf / ((1 + discount_rate) ** (i + 1))
        for i, fcf in enumerate(fcfs)
    ]

    # Terminal Value
    terminal_fcf = fcfs[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    enterprise_value = sum(pv_fcfs) + pv_terminal
    equity_value = enterprise_value - debt + cash
    intrinsic_value_per_share = equity_value / shares

    return {
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "intrinsic_value_per_share": round(intrinsic_value_per_share, 2)
    }
