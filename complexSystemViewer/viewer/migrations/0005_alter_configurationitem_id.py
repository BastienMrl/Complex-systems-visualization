# Generated by Django 5.0.1 on 2024-02-19 10:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0004_alter_configurationitem_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='configurationitem',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
    ]
