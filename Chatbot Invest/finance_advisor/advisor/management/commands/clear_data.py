from django.core.management.base import BaseCommand
from advisor.models import FinancialProfile, InvestmentAdvice, ChatMessage
from django.contrib.auth.models import User

class Command(BaseCommand):
    help = 'Deletes all entries from the database tables'

    def add_arguments(self, parser):
        parser.add_argument(
            '--all',
            action='store_true',
            help='Delete all data including users',
        )

    def handle(self, *args, **options):
        # Delete all chat messages
        chat_count = ChatMessage.objects.count()
        ChatMessage.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f'Successfully deleted {chat_count} chat messages'))

        # Delete all investment advice
        advice_count = InvestmentAdvice.objects.count()
        InvestmentAdvice.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f'Successfully deleted {advice_count} investment advice entries'))

        # Delete all financial profiles
        profile_count = FinancialProfile.objects.count()
        FinancialProfile.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(f'Successfully deleted {profile_count} financial profiles'))

        # If --all flag is provided, delete all users except superusers
        if options['all']:
            user_count = User.objects.filter(is_superuser=False).count()
            User.objects.filter(is_superuser=False).delete()
            self.stdout.write(self.style.SUCCESS(f'Successfully deleted {user_count} users'))
