from django.apps import AppConfig
import subprocess


class ViewerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'viewer'

    def ready(self):
        subprocess.run(["tsc"])
