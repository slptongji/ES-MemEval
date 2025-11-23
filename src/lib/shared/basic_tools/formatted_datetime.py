import datetime


class FormattedDatetime:
    @staticmethod
    def now(format: str = "%Y-%m-%d_%H-%M-%S"):
        return datetime.datetime.now().strftime(format)