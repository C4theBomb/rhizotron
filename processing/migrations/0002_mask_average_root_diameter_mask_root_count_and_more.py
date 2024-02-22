# Generated by Django 5.0.1 on 2024-02-21 19:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processing', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='mask',
            name='average_root_diameter',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='mask',
            name='root_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='mask',
            name='total_root_area',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='mask',
            name='total_root_length',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='mask',
            name='total_root_volume',
            field=models.FloatField(default=0),
        ),
    ]