# Generated by Django 5.0.1 on 2024-02-08 14:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0005_alifemodel_configurationitem_alifemodel'),
    ]

    operations = [
        migrations.AlterField(
            model_name='selectionparameter',
            name='options',
            field=models.CharField(help_text="Wrote options separate with '/'", max_length=512),
        ),
    ]