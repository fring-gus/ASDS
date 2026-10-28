import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration."""
    # SECRET_KEY: MUST be set via .env or environment variable in production.
    # A random fallback is used ONLY for local development.
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-only-change-me-in-production')

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,    # verify connections before using them
    }

    # File uploads
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB — supports up to 120 manual-compare documents

    # Session security
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SESSION_COOKIE_SECURE = os.environ.get('SECURE_COOKIES', 'True').lower() in ['true', '1', 't']      # only send cookies over HTTPS


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}

