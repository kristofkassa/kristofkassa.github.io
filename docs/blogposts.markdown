---
layout: page
title: Blogposts
permalink: /blogposts/
---

Here are all my blogposts, grouped by category:

{% for category in site.categories %}
    <h2>{{ category | first }}</h2>
    {% for post in category.last %}
        <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
        <p>{{ post.excerpt }}</p>
    {% endfor %}
{% endfor %}