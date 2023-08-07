---
layout: page
title: Posts on Derivative Pricing
permalink: /derivative-pricing/
---

Derivatives are financial contracts that derive their value from an underlying asset. These could be stocks, indices, commodities, currencies, exchange rates, or the rate of interest. These financial instruments help you make profits by betting on the future value of the underlying asset. They are complex financial instruments that are used for various purposes including hedging, access to inaccessible markets or commodities, and generating leverage.

{% for post in site.categories['Derivative Pricing'] %}
- [{{ post.title }}]({{ post.url }})
{% endfor %}
