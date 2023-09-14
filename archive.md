---
layout: default
title: Archive
---

#  Welcome to Bearnardd's Log

{% assign postsByYearMonth = site.posts | group_by_exp: "post", "post.date | date: '%B %Y'" %}

<!-- <ul> -->
<!-- {% for yearMonth in postsByYearMonth %} -->
  <!-- <h2>{{ yearMonth.name }}</h2> -->
  <!-- <!-- {% for post in yearMonth.items %} -->
  <!-- <li><a class="post-link" href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li> -->
<!-- {% endfor %} -->
<!-- {% endfor %} -->
<!-- </ul> -->

<ul class="custom-list">
  {% for yearMonth in postsByYearMonth %}
    <!-- <h2>{{ yearMonth.name }}</h2> -->
    {% for post in yearMonth.items %}
      <li class="custom-list-item" >
        {% assign date_format = site.minima.date_format | default: "%b %-d, %Y" %}
        <span class="post-meta">
          <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
          <!-- <p>Tags: 
            {% for tag in post.tags %}
              {% capture tag_name %}{{ tag }}{% endcapture %}
              <a class="post-tag"><nobr>{{ tag_name }}</nobr>&nbsp;</a>
            {% endfor %}
          </p> -->
          <a class="post-desc" href="{{ site.baseurl }}{{ post.url }}">{{ post.desc }}</a>
          <a class="post-footer" href="{{ site.baseurl }}{{ post.url }}"> {{ post.date | date: date_format }} &middot; {{post.author}}</a>
        </span>
      </li>
    {% endfor %}
  {% endfor %}
</ul>

<!-- {% assign postsByYearMonth = site.posts | group_by_exp: "post", "post.date | date: '%B %Y'" %}
{% for yearMonth in postsByYearMonth %}

  <h2>{{ yearMonth.name }}</h2>
  <ul>
    {% for post in yearMonth.items %}
      <li><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %} -->
