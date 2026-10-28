import os
import logging
from flask import Flask, jsonify
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from config import config
from app.models import db, User


login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'error'

csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, default_limits=[])


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def create_app(config_name='default'):
    """Application factory."""
    app = Flask(__name__)
    
    # ── Production Proxy Support ──
    # Trust reverse proxy headers (e.g., from NGINX) to enable correct IP-based rate limiting
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    app.config.from_object(config[config_name])

    # ── Logging ──
    log_level = logging.DEBUG if app.config.get('DEBUG') else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        handlers=[
            logging.StreamHandler(),                          # console
            logging.FileHandler('asds.log', encoding='utf-8'),  # file
        ]
    )
    logger = logging.getLogger('asds')
    logger.info(f'Starting ASDS in {config_name} mode')

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    with app.app_context():
        db.create_all()

    # Register blueprints
    from app.routes import main
    from app.auth import auth
    from app.faculty import faculty
    from app.student import student
    from app.admin import admin_bp
    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(faculty)
    app.register_blueprint(student)
    app.register_blueprint(admin_bp)

    # Rate limit auth endpoints (5 requests/minute per IP)
    auth_limit = limiter.shared_limit("5 per minute", scope="auth")
    app.view_functions['auth.login'] = auth_limit(app.view_functions['auth.login'])
    app.view_functions['auth.register'] = auth_limit(app.view_functions['auth.register'])
    app.view_functions['admin.login'] = auth_limit(app.view_functions['admin.login'])

    # ── Error handlers ──
    @app.errorhandler(413)
    def request_too_large(e):
        """Return JSON for 413 so AJAX callers get a readable message."""
        return jsonify({'error': 'Upload too large. Maximum allowed size is 500 MB.'}), 413

    # ── Health check endpoint ──
    @app.route('/health')
    def health_check():
        """Health check for deployment platforms (Render, Railway, etc.)."""
        return jsonify({'status': 'ok', 'service': 'ASDS'}), 200

    return app
