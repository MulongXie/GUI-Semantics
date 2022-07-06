import cv2


class ElementAttributes:
    def __init__(self):
        # for non-text
        self.element_class = None
        self.icon_class = None
        self.image_class = None
        self.clickable = False
        # for text
        self.text_content = None
        self.text_ner = None
        self.text_bold = False


class BoundingBox:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.width = right - left
        self.height = bottom - top
        self.area = self.width * self.height


class Element:
    def __init__(self, id, left, top, right, bottom):
        self.id = id
        self.attributes = ElementAttributes()
        self.bounding = BoundingBox(left, top, right, bottom)

    def draw_element(self, img, color, show=False):
        cv2.rectangle(img, (self.bounding.left, self.bounding.top), (self.bounding.right, self.bounding.bottom), color, 2)
        if show:
            cv2.imshow('element', img)
            cv2.waitKey()
            cv2.destroyWindow('element')
