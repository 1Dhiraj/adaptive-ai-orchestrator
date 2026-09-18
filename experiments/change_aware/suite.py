"""Workflow suite: 10 workflows in five domains, five requirement changes each
(root, mid-graph, leaf, cosmetic, hidden-dependency).

``uses`` on each step is the authors' label of which requirement facts the
step's task genuinely depends on; the ground-truth must-re-run set of a change
is derived from these labels (see method.must_rerun).
"""
from method import Change, Fact, Step, Workflow


def F(key, value, owner, *aliases):
    return Fact(key, value, owner, list(aliases))


def S(id, role, task, out_type, deps, uses):
    return Step(id, role, task, out_type, list(deps), set(uses))


def C(id, category, request, key, new_value):
    return Change(id, category, request, key, new_value)


SUITE = [
    # ------------------------------------------------------------------ software development
    Workflow(
        "w01", "Software development", "Task manager web application",
        facts={f.key: f for f in [
            F("app.name", "TaskHive", "spec", "taskhive"),
            F("api.style", "REST", "spec", "restful"),
            F("ui.framework", "React", "spec", "jsx", "usestate"),
            F("database.engine", "MongoDB", "schema", "mongodb", "mongoose", "pymongo"),
            F("task.priorities", "low, medium, high", "schema"),
            F("comment.language", "US English", "schema"),
            F("test.framework", "pytest", "tests"),
        ]},
        steps=[
            S("spec", "product manager", "Write a short product specification: purpose, main features and the API style.", "text", [], ["app.name", "api.style", "ui.framework"]),
            S("schema", "database engineer", "Define the data model for tasks and users as code for the chosen database, with brief code comments.", "code", ["spec"], ["database.engine", "task.priorities", "comment.language"]),
            S("backend", "backend developer", "Write Python route handlers that create, list and complete tasks using the chosen database client.", "code", ["spec"], ["database.engine", "api.style"]),
            S("frontend", "frontend developer", "Write the UI component that lists tasks and calls the backend API.", "code", ["spec"], ["ui.framework", "api.style", "app.name"]),
            S("tests", "QA engineer", "Write an integration test plan covering the data model, the backend handlers and the UI component, naming the test framework.", "text", ["schema", "backend", "frontend"], ["test.framework"]),
        ],
        changes=[
            C("w01-root", "root", "Rename the product from TaskHive to PlanPilot everywhere.", "app.name", "PlanPilot"),
            C("w01-mid", "mid", "Tasks need four priority levels: low, medium, high and urgent.", "task.priorities", "low, medium, high, urgent"),
            C("w01-leaf", "leaf", "Use unittest instead of pytest for the tests.", "test.framework", "unittest"),
            C("w01-cosm", "cosmetic", "Code comments should be written in UK English instead of US English.", "comment.language", "UK English"),
            C("w01-hidden", "hidden", "Use PostgreSQL instead of MongoDB.", "database.engine", "PostgreSQL"),
        ],
    ),
    Workflow(
        "w02", "Software development", "Weather alert microservice",
        facts={f.key: f for f in [
            F("service.name", "StormWatch", "brief", "stormwatch"),
            F("api.framework", "FastAPI", "brief", "fastapi"),
            F("alert.threshold", "wind speed above 60 km/h", "rules", "60 km/h", "60km/h"),
            F("docstring.style", "Google style", "rules"),
            F("message.queue", "RabbitMQ", "consumer", "rabbitmq", "amqp", "pika"),
            F("test.tool", "pytest", "tests"),
        ]},
        steps=[
            S("brief", "technical lead", "Write a short service brief: what the service does and how its HTTP API is built.", "text", [], ["service.name", "api.framework"]),
            S("rules", "backend developer", "Write a Python function that decides whether a weather reading triggers an alert, with a docstring.", "code", ["brief"], ["alert.threshold", "docstring.style"]),
            S("consumer", "backend developer", "Write a Python worker that reads weather readings from the message queue.", "code", ["brief"], ["message.queue"]),
            S("deploy", "DevOps engineer", "Write a docker-compose file that runs the service together with its message queue.", "code", ["brief"], ["message.queue", "service.name"]),
            S("tests", "QA engineer", "Write a test plan for the alert rules, the queue worker and the deployment, naming the test tool.", "text", ["rules", "consumer", "deploy"], ["test.tool"]),
        ],
        changes=[
            C("w02-root", "root", "Rename the service from StormWatch to GaleGuard.", "service.name", "GaleGuard"),
            C("w02-mid", "mid", "Only alert when wind speed is above 80 km/h.", "alert.threshold", "wind speed above 80 km/h"),
            C("w02-leaf", "leaf", "Switch the test tool from pytest to unittest.", "test.tool", "unittest"),
            C("w02-cosm", "cosmetic", "Write docstrings in NumPy style instead of Google style.", "docstring.style", "NumPy style"),
            C("w02-hidden", "hidden", "Replace RabbitMQ with Apache Kafka as the message queue.", "message.queue", "Apache Kafka"),
        ],
    ),
    # ------------------------------------------------------------------ data analysis
    Workflow(
        "w03", "Data analysis", "Quarterly sales report",
        facts={f.key: f for f in [
            F("company.name", "Northwind Traders", "brief", "northwind"),
            F("data.file", "sales_2026_q3.csv", "clean"),
            F("currency", "US dollars", "clean", "usd", "us dollar"),
            F("chart.colors", "blue tones", "chart"),
            F("report.length", "one page", "report", "one-page"),
        ]},
        steps=[
            S("brief", "analytics lead", "Write a short analysis brief: which company, which quarter and what questions the report answers.", "text", [], ["company.name"]),
            S("clean", "data engineer", "Write pandas code that loads the sales file and cleans it, converting amounts to the reporting currency.", "code", ["brief"], ["data.file", "currency"]),
            S("metrics", "data analyst", "Write pandas code that computes revenue per region and month, formatted in the reporting currency.", "code", ["brief"], ["currency"]),
            S("chart", "data visualisation specialist", "Write matplotlib code for a bar chart of revenue per region using the requested colours.", "code", ["brief"], ["chart.colors"]),
            S("report", "business analyst", "Write the report text summarising the cleaning, metrics and chart for managers, respecting the requested length.", "text", ["clean", "metrics", "chart"], ["report.length", "company.name"]),
        ],
        changes=[
            C("w03-root", "root", "The report is for Contoso Retail, not Northwind Traders.", "company.name", "Contoso Retail"),
            C("w03-mid", "mid", "Load sales_2026_q3_final.csv instead of sales_2026_q3.csv.", "data.file", "sales_2026_q3_final.csv"),
            C("w03-leaf", "leaf", "The report can be two pages long instead of one.", "report.length", "two pages"),
            C("w03-cosm", "cosmetic", "Use green tones for the chart instead of blue tones.", "chart.colors", "green tones"),
            C("w03-hidden", "hidden", "Report all amounts in euros instead of US dollars.", "currency", "euros"),
        ],
    ),
    Workflow(
        "w04", "Data analysis", "Customer churn analysis",
        facts={f.key: f for f in [
            F("dataset.name", "TelcoCustomers", "brief", "telcocustomers", "telco customers"),
            F("churn.window", "60 days of inactivity", "features", "60 days", "60-day", "60 day"),
            F("model.type", "logistic regression", "model", "logisticregression"),
            F("plot.style", "seaborn whitegrid", "evaluation", "whitegrid"),
            F("summary.format", "bullet list", "summary", "bullet points"),
        ]},
        steps=[
            S("brief", "analytics lead", "Write a short brief for a churn analysis on the named dataset.", "text", [], ["dataset.name"]),
            S("features", "data engineer", "Write pandas code that builds features and a churn label using the churn definition.", "code", ["brief"], ["churn.window"]),
            S("model", "machine learning engineer", "Write scikit-learn code that trains the requested model type on the features.", "code", ["brief"], ["model.type"]),
            S("evaluation", "data scientist", "Write code that evaluates churn predictions against the churn definition and plots results in the requested style.", "code", ["brief"], ["churn.window", "plot.style"]),
            S("summary", "business analyst", "Summarise the feature building, model and evaluation for managers in the requested format.", "text", ["features", "model", "evaluation"], ["summary.format", "dataset.name"]),
        ],
        changes=[
            C("w04-root", "root", "Analyse the StreamFlixSubscribers dataset instead of TelcoCustomers.", "dataset.name", "StreamFlixSubscribers"),
            C("w04-mid", "mid", "Use a random forest instead of logistic regression.", "model.type", "random forest"),
            C("w04-leaf", "leaf", "Write the summary as one short paragraph instead of a bullet list.", "summary.format", "one short paragraph"),
            C("w04-cosm", "cosmetic", "Use the ggplot plot style instead of seaborn whitegrid.", "plot.style", "ggplot"),
            C("w04-hidden", "hidden", "A customer counts as churned after 90 days of inactivity, not 60.", "churn.window", "90 days of inactivity"),
        ],
    ),
    # ------------------------------------------------------------------ web research and messaging
    Workflow(
        "w05", "Web research and messaging", "Product launch announcement",
        facts={f.key: f for f in [
            F("product.name", "Aurora Earbuds", "brief", "aurora"),
            F("price", "129 US dollars", "press", "$129", "129"),
            F("launch.date", "12 October 2026", "press", "october 12", "oct 12", "12 oct"),
            F("social.emoji", "no emojis", "social"),
            F("audience", "existing customers", "email"),
            F("digest.length", "three bullet points", "digest", "3 bullet"),
        ]},
        steps=[
            S("brief", "marketing lead", "Write a launch brief summarising the product, its key features and the launch goal.", "text", [], ["product.name"]),
            S("press", "PR writer", "Write a short press release with the price and launch date.", "text", ["brief"], ["price", "launch.date", "product.name"]),
            S("social", "social media manager", "Write two short social media posts announcing the launch date, following the emoji rule.", "text", ["brief"], ["launch.date", "social.emoji", "product.name"]),
            S("email", "email marketer", "Draft a launch email for the target audience that mentions when the product launches.", "text", ["brief"], ["launch.date", "audience", "product.name"]),
            S("digest", "marketing assistant", "Write an internal digest for the manager summarising the press release, posts and email in the requested length.", "text", ["press", "social", "email"], ["digest.length"]),
        ],
        changes=[
            C("w05-root", "root", "The product is now called Nimbus Earbuds.", "product.name", "Nimbus Earbuds"),
            C("w05-mid", "mid", "The price is 149 US dollars instead of 129.", "price", "149 US dollars"),
            C("w05-leaf", "leaf", "Make the internal digest five bullet points.", "digest.length", "five bullet points"),
            C("w05-cosm", "cosmetic", "Social posts may use one emoji each.", "social.emoji", "one emoji per post"),
            C("w05-hidden", "hidden", "The launch moves to 19 October 2026.", "launch.date", "19 October 2026"),
        ],
    ),
    Workflow(
        "w06", "Web research and messaging", "Research digest on heat pumps",
        facts={f.key: f for f in [
            F("article.topic", "heat pumps in cold climates", "brief", "heat pump"),
            F("key.statistic", "heat pumps cut heating emissions by 40 percent", "summary", "40 percent", "40%"),
            F("summary.words", "120 words", "summary"),
            F("audience", "city council members", "faq", "council"),
            F("slack.format", "plain text", "slack"),
            F("signoff", "Regards, Research Team", "email", "research team"),
        ]},
        steps=[
            S("brief", "research lead", "Write a research brief describing the article topic and what the team should extract.", "text", [], ["article.topic"]),
            S("summary", "research analyst", "Summarise the article findings including the key statistic, within the word limit.", "text", ["brief"], ["key.statistic", "summary.words", "article.topic"]),
            S("faq", "policy writer", "Write three FAQ entries for the audience, quoting the key statistic.", "text", ["brief"], ["key.statistic", "audience", "article.topic"]),
            S("slack", "communications officer", "Write a short Slack message telling the team the research digest is ready, in the requested format.", "text", ["brief"], ["slack.format", "article.topic"]),
            S("email", "communications officer", "Write an email to stakeholders that combines the summary, FAQ and Slack note, ending with the sign-off.", "text", ["summary", "faq", "slack"], ["signoff"]),
        ],
        changes=[
            C("w06-root", "root", "The digest is about district heating networks, not heat pumps.", "article.topic", "district heating networks"),
            C("w06-mid", "mid", "Keep the summary to 80 words.", "summary.words", "80 words"),
            C("w06-leaf", "leaf", "Sign the email 'Thanks, Policy Desk'.", "signoff", "Thanks, Policy Desk"),
            C("w06-cosm", "cosmetic", "Format the Slack message with bold headings instead of plain text.", "slack.format", "bold headings"),
            C("w06-hidden", "hidden", "Correction: the article says emissions fall by 55 percent.", "key.statistic", "heat pumps cut heating emissions by 55 percent"),
        ],
    ),
    # ------------------------------------------------------------------ document processing
    Workflow(
        "w07", "Document processing", "Contract obligation extraction",
        facts={f.key: f for f in [
            F("contract.party", "Acme Logistics", "brief", "acme"),
            F("notice.period", "30 days", "extract", "thirty days", "30-day"),
            F("jurisdiction", "State of New York", "review", "new york"),
            F("risk.labels", "low, medium, high", "risk"),
            F("memo.length", "150 words", "memo"),
        ]},
        steps=[
            S("brief", "legal operations lead", "Write a brief describing the supply contract with the counterparty and what must be extracted.", "text", [], ["contract.party"]),
            S("extract", "contract analyst", "Extract the key obligations, including the termination notice period, as a JSON list.", "json", ["brief"], ["notice.period", "contract.party"]),
            S("review", "commercial lawyer", "Review the governing-law and dispute clauses under the jurisdiction.", "text", ["brief"], ["jurisdiction"]),
            S("risk", "risk analyst", "Produce a JSON risk table for the contract clauses, considering the jurisdiction and using the risk labels.", "json", ["brief"], ["jurisdiction", "risk.labels"]),
            S("memo", "legal counsel", "Write a memo to management combining the obligations, clause review and risk table, within the length limit.", "text", ["extract", "review", "risk"], ["memo.length", "contract.party"]),
        ],
        changes=[
            C("w07-root", "root", "The counterparty is Borealis Freight, not Acme Logistics.", "contract.party", "Borealis Freight"),
            C("w07-mid", "mid", "The termination notice period is 45 days.", "notice.period", "45 days"),
            C("w07-leaf", "leaf", "The memo may be up to 250 words.", "memo.length", "250 words"),
            C("w07-cosm", "cosmetic", "Use the risk labels L, M and H instead of low, medium, high.", "risk.labels", "L, M, H"),
            C("w07-hidden", "hidden", "The contract is governed by Delaware law, not New York.", "jurisdiction", "State of Delaware"),
        ],
    ),
    Workflow(
        "w08", "Document processing", "Remote work policy summary",
        facts={f.key: f for f in [
            F("org.name", "Greenfield Hospital", "brief", "greenfield"),
            F("eligibility", "employees with 6 months of service", "points", "6 months", "six months"),
            F("effective.date", "1 January 2027", "points", "january 2027"),
            F("review.cycle", "reviewed annually", "points", "annually", "annual review"),
            F("checklist.style", "numbered list", "checklist"),
            F("notice.tone", "formal", "notice"),
        ]},
        steps=[
            S("brief", "HR lead", "Write a brief explaining which organisation's remote work policy is being summarised and for whom.", "text", [], ["org.name"]),
            S("points", "HR policy analyst", "List the key points of the policy: eligibility, start date and how often it is reviewed.", "text", ["brief"], ["eligibility", "effective.date", "review.cycle"]),
            S("faq", "HR advisor", "Write three FAQ entries about who can work remotely and from when.", "text", ["brief"], ["eligibility", "effective.date"]),
            S("checklist", "HR coordinator", "Produce a JSON checklist of steps an employee follows to request remote work, in the requested style.", "json", ["brief"], ["checklist.style"]),
            S("notice", "internal communications writer", "Write the staff notice combining key points, FAQ and checklist in the requested tone.", "text", ["points", "faq", "checklist"], ["notice.tone", "org.name"]),
        ],
        changes=[
            C("w08-root", "root", "The policy belongs to Riverside Clinic, not Greenfield Hospital.", "org.name", "Riverside Clinic"),
            C("w08-mid", "mid", "The policy will be reviewed every six months instead of annually.", "review.cycle", "reviewed every six months"),
            C("w08-leaf", "leaf", "Make the staff notice friendly rather than formal.", "notice.tone", "friendly"),
            C("w08-cosm", "cosmetic", "Show the checklist as checkboxes instead of a numbered list.", "checklist.style", "checkbox list"),
            C("w08-hidden", "hidden", "Employees become eligible after 3 months of service.", "eligibility", "employees with 3 months of service"),
        ],
    ),
    # ------------------------------------------------------------------ DevOps
    Workflow(
        "w09", "DevOps", "CI pipeline for a Python library",
        facts={f.key: f for f in [
            F("repo.name", "fastgeo", "brief"),
            F("python.version", "Python 3.11", "ci", "3.11", "python:3.11"),
            F("test.command", "pytest -q", "ci"),
            F("config.comments", "no comments", "lint"),
            F("pr.template", "short summary", "pr"),
        ]},
        steps=[
            S("brief", "engineering lead", "Write a short brief for adding continuous integration to the repository.", "text", [], ["repo.name"]),
            S("ci", "DevOps engineer", "Write a GitHub Actions workflow that installs the Python version and runs the test command.", "code", ["brief"], ["python.version", "test.command"]),
            S("lint", "DevOps engineer", "Write a flake8 configuration file for the repository, following the comment rule.", "code", ["brief"], ["config.comments"]),
            S("docker", "DevOps engineer", "Write a Dockerfile for running the library's tests on the Python version.", "code", ["brief"], ["python.version"]),
            S("pr", "software engineer", "Write the pull request description for the CI workflow, lint config and Dockerfile using the PR template.", "text", ["ci", "lint", "docker"], ["pr.template", "repo.name"]),
        ],
        changes=[
            C("w09-root", "root", "The repository is called geoquick now, not fastgeo.", "repo.name", "geoquick"),
            C("w09-mid", "mid", "Run the tests with coverage: pytest -q --cov=fastgeo.", "test.command", "pytest -q --cov=fastgeo"),
            C("w09-leaf", "leaf", "Use a PR template with a summary plus a checklist.", "pr.template", "summary plus checklist"),
            C("w09-cosm", "cosmetic", "Add brief comments to the lint configuration.", "config.comments", "brief comments"),
            C("w09-hidden", "hidden", "Target Python 3.12 instead of 3.11.", "python.version", "Python 3.12"),
        ],
    ),
    Workflow(
        "w10", "DevOps", "Kubernetes deployment for an orders API",
        facts={f.key: f for f in [
            F("service.name", "orders-api", "brief"),
            F("container.port", "port 8080", "deployment", "8080"),
            F("replicas", "3 replicas", "deployment", "replicas: 3"),
            F("latency.alert", "p95 latency above 300 ms", "alerts", "300ms", "300 ms"),
            F("yaml.comments", "no comments", "alerts"),
            F("runbook.audience", "on-call engineers", "runbook", "on-call"),
        ]},
        steps=[
            S("brief", "platform lead", "Write a short brief for deploying the service to Kubernetes.", "text", [], ["service.name"]),
            S("deployment", "platform engineer", "Write the Kubernetes Deployment manifest with the container port and replica count.", "code", ["brief"], ["container.port", "replicas", "service.name"]),
            S("service", "platform engineer", "Write the Kubernetes Service manifest that routes traffic to the container port.", "code", ["brief"], ["container.port", "service.name"]),
            S("alerts", "site reliability engineer", "Write Prometheus alerting rules for the latency threshold, following the comment rule.", "code", ["brief"], ["latency.alert", "yaml.comments"]),
            S("runbook", "site reliability engineer", "Write a runbook for the audience covering the deployment, service and alerts.", "text", ["deployment", "service", "alerts"], ["runbook.audience", "service.name"]),
        ],
        changes=[
            C("w10-root", "root", "Rename the service from orders-api to payments-api.", "service.name", "payments-api"),
            C("w10-mid", "mid", "Run 5 replicas instead of 3.", "replicas", "5 replicas"),
            C("w10-leaf", "leaf", "Write the runbook for new team members instead of on-call engineers.", "runbook.audience", "new team members"),
            C("w10-cosm", "cosmetic", "Add short comments to the alerting rules YAML.", "yaml.comments", "short comments"),
            C("w10-hidden", "hidden", "The container now listens on port 9090.", "container.port", "port 9090"),
        ],
    ),
]
