# Generated by Django 5.0.1 on 2024-02-19 16:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0009_alter_configurationitem_idhtml'),
    ]

    operations = [
        migrations.RenameField(
            model_name='parameter',
            old_name='idHtml',
            new_name='classHtml',
        ),
    ]