import typing

from cmenpy.decoder.strings import StringDecoder


class CategoricalDecoder(StringDecoder):
    def generate(self) -> typing.List[typing.Any]:
        return [self.generator.choice(arr) for arr in self.labels]
