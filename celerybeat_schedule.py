from celery import Celery
from celery.schedules import crontab

app = Celery('advanced_trading_app')
app.config_from_object('celery_config')

app.conf.beat_schedule = {
    'update-all-data-every-hour': {
        'task': 'celery_tasks.update_all_data',
        'schedule': crontab(minute=0),  # Run every hour
    },
    'retrain-models-daily': {
        'task': 'celery_tasks.daily_model_retraining',
        'schedule': crontab(hour=0, minute=0),  # Run at midnight
    },
}

