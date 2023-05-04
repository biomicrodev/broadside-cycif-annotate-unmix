from dataclasses import dataclass


@dataclass
class Point2D:
    """
    I like being strict about the dunder methods.
    """

    x: float
    y: float

    def __add__(self, other: "Point2D"):
        return Point2D(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point2D"):
        return Point2D(x=self.x - other.x, y=self.y - other.y)

    def __truediv__(self, other):
        if isinstance(other, Point2D):
            return Point2D(x=self.x / other.x, y=self.y / other.y)
        else:
            return Point2D(x=self.x / other, y=self.y / other)

    def as_dict(self):
        return dict(x=self.x, y=self.y)
