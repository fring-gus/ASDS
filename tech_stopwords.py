"""
ASDS — Technical & Academic Stopwords
─────────────────────────────────────
Curated terms to strip from CS/Engineering student project abstracts
so the similarity engine focuses on the *problem domain*, not the
implementation stack.

Categories:
  1. Programming Languages
  2. Web Frameworks & Libraries
  3. Mobile & Desktop Platforms
  4. Databases & Data Stores
  5. Cloud, DevOps & Infrastructure
  6. AI / ML / Data Science
  7. Networking, Protocols & Formats
  8. Architecture & Design Patterns
  9. Generic CS / Software Engineering Terms
  10. Academic / Report Filler Words
"""

TECH_STOPWORDS = {
    # ── 1. Programming Languages ──────────────────────────
    'python', 'java', 'javascript', 'typescript', 'cpp', 'cplusplus',
    'csharp', 'ruby', 'php', 'swift', 'kotlin', 'rust', 'golang',
    'scala', 'perl', 'matlab', 'dart', 'lua', 'haskell', 'elixir',
    'clojure', 'fortran', 'cobol', 'groovy', 'objectivec', 'assembly',
    'vhdl', 'verilog', 'prolog', 'lisp', 'erlang', 'julia', 'ocaml',
    'solidity', 'sql', 'plsql', 'nosql', 'bash', 'powershell', 'shell',

    # ── 2. Web Frameworks & Libraries ─────────────────────
    'react', 'reactjs', 'angular', 'angularjs', 'vue', 'vuejs',
    'svelte', 'nextjs', 'nuxtjs', 'gatsby', 'remix',
    'django', 'flask', 'fastapi', 'spring', 'springboot',
    'express', 'expressjs', 'nestjs', 'koa', 'hapi',
    'laravel', 'symfony', 'codeigniter', 'cakephp',
    'rails', 'sinatra', 'aspnet', 'dotnet', 'blazor',
    'jquery', 'bootstrap', 'tailwind', 'tailwindcss',
    'materialui', 'chakraui', 'antdesign',
    'webpack', 'vite', 'parcel', 'rollup', 'babel', 'eslint',
    'npm', 'yarn', 'pnpm', 'pip', 'conda', 'maven', 'gradle',
    'nodejs', 'node', 'deno', 'bun',

    # ── 3. Mobile & Desktop Platforms ─────────────────────
    'android', 'ios', 'flutter', 'reactnative', 'xamarin',
    'ionic', 'cordova', 'phonegap', 'swiftui', 'jetpack',
    'compose', 'electron', 'tauri', 'wpf', 'winforms',
    'qt', 'gtk', 'tkinter', 'kivy',
    'xcode', 'androidstudio', 'gradle',

    # ── 4. Databases & Data Stores ────────────────────────
    'mysql', 'postgresql', 'postgres', 'sqlite', 'mariadb',
    'oracle', 'mssql', 'sqlserver',
    'mongodb', 'mongoose', 'couchdb', 'couchbase',
    'redis', 'memcached', 'cassandra', 'dynamodb',
    'firebase', 'firestore', 'supabase', 'appwrite',
    'elasticsearch', 'solr', 'neo4j', 'graphdb',
    'influxdb', 'timescaledb', 'cockroachdb',
    'orm', 'sequelize', 'prisma', 'typeorm', 'hibernate',
    'sqlalchemy', 'knex', 'drizzle',

    # ── 5. Cloud, DevOps & Infrastructure ─────────────────
    'aws', 'azure', 'gcp', 'googlecloud', 'digitalocean',
    'heroku', 'netlify', 'vercel', 'railway', 'render',
    'docker', 'dockerfile', 'kubernetes', 'k8s', 'helm',
    'terraform', 'ansible', 'puppet', 'chef', 'vagrant',
    'jenkins', 'githubactions', 'gitlab', 'circleci', 'travisci',
    'nginx', 'apache', 'caddy', 'iis',
    'linux', 'ubuntu', 'centos', 'debian', 'redhat', 'windows',
    'macos', 'unix', 'virtualbox', 'vmware',
    'cicd', 'devops', 'mlops', 'gitops',
    'ec2', 's3', 'lambda', 'ecs', 'fargate', 'cloudfront',
    'loadbalancer', 'cdn', 'ssl', 'tls', 'dns',
    'serverless', 'paas', 'saas', 'iaas', 'baas',

    # ── 6. AI / ML / Data Science ─────────────────────────
    'tensorflow', 'pytorch', 'keras', 'scikitlearn', 'sklearn',
    'opencv', 'pandas', 'numpy', 'scipy', 'matplotlib',
    'seaborn', 'plotly', 'huggingface', 'transformers',
    'spacy', 'nltk', 'gensim', 'fasttext', 'word2vec',
    'chatgpt', 'openai', 'llms',
    'cnn', 'rnn', 'lstm', 'gan', 'vae', 'autoencoder',
    'resnet', 'vgg', 'yolo', 'ssd', 'unet', 'alexnet',
    'inception', 'mobilenet', 'efficientnet',
    'xgboost', 'lightgbm', 'catboost', 'randomforest',
    'svm', 'knn', 'kmeans', 'dbscan', 'pca',
    'neuralnetwork', 'deeplearning', 'machinelearning',
    'supervisedlearning', 'unsupervisedlearning',
    'backpropagation', 'gradientdescent', 'epoch', 'epochs',
    'overfitting', 'underfitting', 'hyperparameter',
    'crossvalidation', 'trainset', 'testset', 'validationset',
    'accuracy', 'precision', 'recall', 'f1score', 'f1',
    'auc', 'roc', 'confusionmatrix',
    'regression',
    'finetuning', 'pretraining',
    'dataset', 'datasets', 'dataframe',
    'featureextraction', 'dimensionalityreduction',
    'computervision',
    'jupyter', 'notebook', 'colab', 'kaggle',

    # ── 7. Networking, Protocols & Formats ────────────────
    'http', 'https', 'tcp', 'udp', 'ip', 'ftp', 'ssh', 'smtp',
    'websocket', 'websockets', 'mqtt', 'amqp', 'grpc',
    'json', 'xml', 'yaml', 'csv', 'html', 'css',
    'rest', 'restful', 'restapi', 'graphql', 'soap', 'rpc',
    'oauth', 'oauth2', 'jwt', 'saml', 'openid',
    'ssl', 'tls', 'cors', 'csrf',
    'ipv4', 'ipv6', 'dhcp', 'nat', 'vpn', 'proxy',

    # ── 8. Architecture & Design Patterns ─────────────────
    'mvc', 'mvvm', 'mvp', 'clean architecture',
    'microservice', 'microservices', 'monolith', 'monolithic',
    'soa', 'eventdriven', 'pubsub', 'messagequeue',
    'singleton', 'factory', 'observer', 'strategy', 'decorator',
    'repository', 'middleware', 'controller', 'router',
    'frontend', 'backend', 'fullstack',
    'crud', 'api', 'apis', 'sdk', 'cli',
    'responsive', 'spa', 'pwa', 'ssr', 'csr', 'ssg',
    'agile', 'scrum', 'kanban', 'waterfall', 'sdlc',
    'uml', 'erdiagram', 'dfd', 'usecase',
    'unittest', 'testing', 'tdd', 'bdd', 'selenium',
    'cypress', 'jest', 'mocha', 'pytest', 'junit',
    'git', 'github', 'bitbucket', 'svn', 'versioncontrol',
    'ide', 'vscode', 'intellij', 'eclipse', 'vim',

    # ── 9. Generic CS / Software Engineering ──────────────
    'software', 'application', 'app', 'apps', 'program',
    'website', 'webpage', 'web', 'mobile', 'desktop',
    'framework', 'frameworks', 'library', 'libraries',
    'tool', 'tools', 'platform', 'platforms',
    'module', 'modules', 'component', 'components',
    'plugin', 'plugins', 'package', 'packages',
    'server', 'client', 'user', 'users', 'admin',
    'interface', 'ui', 'ux', 'gui', 'dashboard',
    'database', 'storage', 'cache', 'caching',
    'deploy', 'deployment', 'deployed', 'hosting', 'hosted',
    'code', 'codebase', 'source', 'sourcecode',
    'script', 'scripting', 'automated',
    'debug', 'debugging', 'debugger',
    'compile', 'compiler', 'interpreter', 'runtime',
    'algorithm', 'algorithms', 'datastructure', 'datastructures',
    'function', 'functions', 'method', 'methods',
    'class', 'classes', 'object', 'objects',
    'variable', 'variables', 'parameter', 'parameters',
    'array', 'list', 'stack', 'queue', 'tree', 'graph',
    'hash', 'hashing', 'decryption',
    'login', 'signup',
    'session', 'cookie', 'cookies',
    'request', 'response', 'endpoint', 'endpoints',
    'query', 'queries', 'schema', 'schemas',
    'table', 'tables', 'column', 'columns', 'row', 'rows',
    'index', 'indexing', 'pagination',
    'upload', 'download', 'fetch', 'fetching',
    'render', 'rendering', 'rendered',
    'parse', 'parsing', 'parser', 'serialize', 'serialization',
    'integrate', 'integration', 'integrated',
    'configure', 'configuration', 'config', 'setup',
    'install', 'installation', 'installed',
    'implement', 'implementation', 'implemented',
    'develop', 'development', 'developer', 'developed',
    'build', 'building', 'built',
    'test', 'tests', 'tested',
    'feature', 'features', 'functionality', 'functionalities',
    'requirement', 'requirements', 'specification', 'specifications',
    'version', 'versioning', 'update', 'updates', 'updated',
    'bug', 'bugs', 'fix', 'fixes', 'fixed', 'patch',
    'optimize', 'optimized', 'performance',
    'scalable', 'scalability', 'reliable', 'reliability',
    'secure', 'security', 'vulnerable', 'vulnerability',
    'responsive', 'compatibility', 'compatible',
    'architecture', 'architectural', 'infrastructure',
    'protocol', 'protocols',
    'operating', 'kernel', 'process', 'thread', 'threads',
    'memory', 'cpu', 'gpu', 'ram', 'disk', 'storage',
    'network', 'networking', 'bandwidth',
    'virtual', 'virtualization', 'container', 'containers',
    'abstraction', 'encapsulation', 'inheritance', 'polymorphism',
    'iteration', 'iterative', 'recursive', 'recursion',
    'binary', 'hexadecimal', 'boolean', 'integer', 'string',
    'pixel', 'pixels', 'resolution',
    'log', 'logs', 'logging', 'monitor', 'monitoring',
    'backup', 'restore', 'migration',
    'workflow', 'pipeline', 'pipelines',
    'template', 'templates', 'boilerplate',
    'documentation', 'docs',
    'open source', 'opensource', 'proprietary',
    'tech', 'technology', 'technologies', 'technical',
    'digital', 'electronic', 'compute', 'computing',
    'data', 'information',

    # ── 10. Academic / Report Filler Words ────────────────
    'project', 'paper', 'thesis', 'dissertation', 'study',
    'research', 'proposed', 'system', 'method', 'methodology',
    'approach', 'framework', 'solution', 'aim', 'aims',
    'objective', 'objectives', 'goal', 'goals', 'scope',
    'overview', 'abstract', 'introduction', 'conclusion',
    'results', 'future', 'work', 'present', 'presents',
    'presented', 'propose', 'proposes', 'demonstrate',
    'demonstrates', 'demonstrated', 'show', 'shows', 'shown',
    'experiment', 'experiments', 'experimental', 'evaluate',
    'evaluation', 'analysis', 'investigate', 'investigation',
}

