# Generated by Django 5.0.1 on 2024-03-11 18:42

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0025_rename_configurationitem_parameter_transformer'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='transformeritem',
            name='aLifeModel',
        ),
    ]