# _config.yml

# 站点基本信息
title: "燃烧的番茄的个人博客"        # 网站标题（浏览器标签、首页大标题）
description: "记录学习、生活与技术探索的小天地"  # 站点描述（SEO 用）
url: "https://jackpetpet.github.io"  # 你的 GitHub Pages 地址（必须正确，否则图片/链接会404）
baseurl: ""                          # 如果部署在子路径（如 https://xxx.github.io/blog），填 "/blog"，否则留空

# 主题与作者
theme: minima                       # GitHub Pages 官方支持的极简主题（可替换成其他主题，如 jekyll-theme-slate）
author:
  name: "燃烧的番茄"                 # 你的名字/昵称
  email: "your-email@example.com"    # 邮箱（非必须，若做评论系统可能用到）

# 时区与语言
timezone: Asia/Shanghai             # 时区（确保文章时间正确）
lang: zh-CN                         # 站点语言（影响日期格式、翻译等）

# 博客文章配置
permalink: /:year/:month/:day/:title/  # 文章链接格式（如 2025/08/24/我的第一篇博客）
markdown: kramdown                   # 解析 Markdown 的引擎（GitHub Pages 推荐）

# 社交链接（可选，加到页脚或导航）
social:
  github: https://github.com/jackpetpet  # 你的 GitHub 主页
  weibo: https://weibo.com/your-account  # 微博等，按需添加