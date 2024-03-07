# Generated by Django 5.0.2 on 2024-03-07 15:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processing', '0003_model'),
    ]

    operations = [
        migrations.AlterField(
            model_name='model',
            name='model_type',
            field=models.CharField(choices=[('unet', 'UNET'), ('resnet101', 'ResNet101'), ('resnet50', 'ResNet50'), ('resnet18', 'ResNet18'), ('resnet34', 'ResNet34'), ('resnet152', 'ResNet152')], default='unet', max_length=50),
        ),
    ]