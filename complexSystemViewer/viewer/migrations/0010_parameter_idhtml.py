# Generated by Django 5.0.1 on 2024-02-14 14:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0009_tool'),
    ]

    operations = [
        migrations.AddField(
            model_name='parameter',
            name='idHtml',
            field=models.CharField(max_length=128, null=True),
        ),
    ]
