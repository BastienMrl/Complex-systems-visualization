# Generated by Django 5.0.1 on 2024-02-28 14:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0018_transformeritem_displayedbydefault_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='transformeritem',
            name='description',
            field=models.CharField(default='Aucune information', max_length=1024),
            preserve_default=False,
        ),
    ]
