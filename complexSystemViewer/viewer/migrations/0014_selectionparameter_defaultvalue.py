# Generated by Django 5.0.1 on 2024-02-19 17:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0013_parameter_paramid'),
    ]

    operations = [
        migrations.AddField(
            model_name='selectionparameter',
            name='defaultValue',
            field=models.SmallIntegerField(default=0),
        ),
    ]
