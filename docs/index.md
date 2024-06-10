<p align="center">
 <img alt="Logo dark mode" src="_static/images/logo-horizontal-transparent.png#only-dark" width="50%" />
 <img alt="Logo light mode" src="_static/images/BAAL-horizontal-logo-black.png#only-light" width="50%"/>
</p>


Baal is a Bayesian active learning library.
We provide methods to estimate sampling from the posterior distribution
in order to maximize the efficiency of labelling during active learning. Our library is suitable for research and industrial applications.

To know more on what is Bayesian active learning, see our [User guide](user_guide/index.md).

We are a member of Pytorch's ecosystem, and we welcome contributions from the community.

!!! tip "Baal 2.0 !"
    Baal is now version 2.0! We made a lot of changes to make everyone life easier!
    See our [Release Note](https://github.com/baal-org/baal/releases/tag/v2.0.0) for details.

::cards:: cols=2
- title: User Guide
  content: |
    Learn how to use Baal
  image: /_static/images/open-book_171322.png
  url: /user_guide
- title: Get Help
  content: |
    Submit an issue on Github
  image: /_static/images/github-mark.svg
  url: https://github.com/baal-org/baal/issues/new/choose
- title: Community
  content: |
    Join our Slack!
  image: https://upload.wikimedia.org/wikipedia/commons/d/d5/Slack_icon_2019.svg
  url: https://join.slack.com/t/baal-world/shared_invite/zt-z0izhn4y-Jt6Zu5dZaV2rsAS9sdISfg
- title: FAQ
  content: Most common questions
  image: /_static/images/help.png
  url: support/faq
::/cards::


## Installation

Baal is available as a package on PyPI:

`pip install baal`

??? "Additional dependencies for vision and NLP"
    
    `baal[nlp]` installs needed dependencies for HuggingFace support.

    `baal[vision]` installs dependencies for our Lightning-Flash integration.
