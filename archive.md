---
layout: default
title: Archive
---

# Archive

Browse all posts by month and year.

<!-- {% assign postsByYearMonth = site.posts | group_by_exp: "post", "post.date | date: '%B %Y'" %}
{% for yearMonth in postsByYearMonth %}

  <h2>{{ yearMonth.name }}</h2>
  <ul>
    {% for post in yearMonth.items %}
      <li><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %} -->

<div class="home">
  <ul class="post-list">
  {% assign postsByYearMonth = site.posts | group_by_exp: "post", "post.date | date: '%B %Y'" %}
  {% for yearMonth in postsByYearMonth %}
  <h2>{{ yearMonth.name }}</h2>
    <ul>
      {% for post in yearMonth.items %}
        <li><a class="post-link" href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
      {% endfor %}
    </ul>
  </ul>
</div>
{% endfor %}
