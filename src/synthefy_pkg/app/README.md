# Synthefy Backend API

## Directory Structure
```
app/
    routers/
        __init__.py
        search.py
        synthesis.py
        ...
    services/
        configs/
            search_config.yaml
            synthesis_config.yaml
            ...
        ...
        __init__.py
        search_service.py
        synthesis_service.py

    tests/
        test_{}_service.py
    utils/
        __init__.py
        utils.py
    __init__.py
    config.py
    data_models.py
    main.py

```

## Setup

### Environment Setup
1. We rely on the same venv as the synthefy-package, so be sure to activate it.
2. Environment Vars:
    - Synthefy package variables:
        - export SYNTHEFY_DATASETS_BASE=/Users/raimi/synthefy_data/synthefy_package_datasets
        - export SYNTHEFY_PACKAGE_BASE=/Users/raimi/synthefy/synthefy-package
    - Service Specific variables - see config.py for the variables used by earch service:
        - Note, the variables are also maintained in services/configs/*_config.yaml so we don't need to mess with environment variables too much. Though, environment variables take precedence over the yaml files.

### Running tests
From app/tests, run
`pytest` to run all tests.
`pytest -s` to run test with printing, IPython Embed mode enabled.

### Running the UI APIs locally:
From `app`, run
- `python main.py --config services/configs/api_config_general_dev.yaml` will start up the API using files from dev aws s3 bucket (uploading/downloading to s3 and cleaning up locally).
- `python main.py --config services/configs/api_config_general_prod.yaml` will start up the API using files from prod aws s3 bucket (uploading/downloading to s3 and cleaning up locally).
- `python main.py --config services/configs/api_config_general_local.yaml` will start up the API that works with local files only.
- navigate to http://0.0.0.0:8000/docs for the API documentation.
- You can test each api endpoint on the docs page, or by hitting the endpoint by going to http://0.0.0.0:8000/api/synthesis/default
    - On the /docs page, you can click the default endpoint to get the request format, then copy paste this into the request for the synthesis.
- You can also run the server locally in hot reload mode (the server will restart when you make changes to the code) using either the configuration file saved in the environment variable`SYNTHEFY_CONFIG_PATH` or the default `api_config_general_local.yaml` with:
```shell
make init-env
# Fill in the variables in .env.local
make run_dev_server
```
    - While you develop in hot-reload mode, it is recommended that you disable auto-save in your IDE to avoid constantly saving the files in the dev server and causing the server to restart.


### Running the streaming APIs locally:
- `python examples/launch_backend.py --config examples/configs/api_configs/api_config_{config_name}.yaml` will start up the API that works with local files only.
- navigate to http://0.0.0.0:8000/docs for the API documentation.
- You can test each api endpoint on the docs page, or by hitting the endpoint by going to http://0.0.0.0:8000/api/synthesis/default
    - On the /docs page, you can click the default endpoint to get the request format, then copy paste this into the request for the synthesis.

### Run DB Migrations

Change env variables as per your local setup requirement:

```shell
cd synthefy-package/src/synthefy_pkg
export SYNTHEFY_PACKAGE_BASE=/home/ubuntu/code/synthefy-package
export SYNTHEFY_DATASETS_BASE=/home/ubuntu/data
export DATABASE_URL='mysql+mysqlconnector://<USERNAME>:<PASSWORD>@localhost/<DB_NAME>'
```
Create and apply migrations:
```shell
alembic revision --autogenerate -m "Create some new table"
alembic upgrade head
```

## Docker
TODO
