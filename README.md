## Histopathological Cancer Detection

*[ TODO Add project description]*

## How to Run

First, make sure that you have [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) installed and install the project.

```shell
mkvirtualenv Histopathological Cancer Detection
pip install -e 'git+https://github.com/cmpatino/Histopathological-Cancer-Detection.git.git#egg=cancer_detection'
```

> For a private repository accessible only through an SSH authentication, substitute `git+https://github.com` with `git+ssh://git@github.com`.

*[ TODO Add instructions to run package scritps ]*

## How to Contribute

First, make sure that you have [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html) installed and install the project in development mode.

```shell
mkvirtualenv Histopathological Cancer Detection
git clone https://github.com/cmpatino/Histopathological-Cancer-Detection.git.git
cd Histopathological Cancer Detection
pip install -r requirements.txt
pip install -e .
pip freeze | grep -v cancer_detection > requirements.txt
```

> For a private repository accessible only through an SSH authentication, substitute `https://github.com/` with `git@github.com:`.

Then, create or select a GitHub branch and have fun... 