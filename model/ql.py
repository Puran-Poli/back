#ql.py
import datetime
import enum
from typing import Union,Optional,List

from pydantic import BaseModel


class Hierarchy(BaseModel):
    gender: str
    vertical: str
    fabric_category: str
    brand: str
    usage: str
    brick: str
    product: str
    sub_product: str
    target_audience: str
    fit: str


class EditHierarchy(BaseModel):
    gender: Optional[str]
    vertical: Optional[str]
    fabric_category: Optional[str]
    brand: Optional[str]
    usage: Optional[str]
    brick: Optional[str]
    product: Optional[str]
    sub_product: Optional[str]
    target_audience: Optional[str]
    fit: Optional[str]


class Colour(BaseModel):
    colour_family: str
    primary_colour: str
    secondary_colour: Union[str, None]
    tertiary_colour: Union[str, None]


class EditColour(BaseModel):
    colour_family: Union[str, None]
    primary_colour: Union[str, None]
    secondary_colour: Union[str, None]
    tertiary_colour: Union[str, None]


class Design(BaseModel):
    garment_pattern: Union[str, None]
    print_pattern_type: Union[str, None]
    number_of_components: Union[str, None]
    number_of_pockets: int = 0
    pocket_type: Union[str, None]
    neck: Union[str, None]
    collar: Union[str, None]
    placket: Union[str, None]
    length: Union[str, None]
    sleeve_length: Union[str, None]
    sleeve_type: Union[str, None]
    hemline: Union[str, None]
    waist_rise: Union[str, None]
    closure: Union[str, None]
    footwear_ankle_type: Union[str, None]
    footwear_insole: Union[str, None]
    dress_type: Union[str, None] #YashBodke
    top_style_type: Union[str, None]
    jacket_type: Union[str, None]
    distress: Union[str, None]
    fade: Union[str, None]
    dungaree_bottom_type: Union[str, None]




class Fabric(BaseModel):
    fabric_code: str
    fabric_rate: Union[int, None]
    fabric_story: Union[str, None]
    fabric_composition: str
    fabric_hsn: Union[int, None]
    fabric_weave_pattern: Union[str, None]
    fabric_vendor: Union[str, None]
    denim_cast: Union[str, None]
    denim_wash: Union[str, None]
    wash_care: Union[str, None]
    footwear_upper_material: Union[str, None]
    footwear_sole_material: Union[str, None]


class ReadFabric(BaseModel):
    fabric_code: Union[str, None]
    fabric_rate: Union[int, None]
    fabric_story: Union[str, None]
    fabric_composition: Union[str, None]
    fabric_hsn: Union[int, None]
    fabric_weave_pattern: Union[str, None]
    fabric_vendor: Union[str, None]
    denim_cast: Union[str, None]
    denim_wash: Union[str, None]
    wash_care: Union[str, None]
    footwear_upper_material: Union[str, None]
    footwear_sole_material: Union[str, None]


class Dates(BaseModel):
    first_grn_date: Union[datetime.date, None]
    first_live_date: Union[datetime.date, None]
    first_sold_date: Union[datetime.date, None]


class DateTypes(enum.Enum):
    updated_at = "updated_at"
    created_at = "created_at"
    first_grn_date = "first_grn_date"
    first_live_date = "first_live_date"
    first_sold_date = "first_sold_date"


class ExportParams(BaseModel):
    date_type: DateTypes
    start_date: datetime.date
    end_date: datetime.date
    status: str = "Approved"


class Size(BaseModel):
    barcode_size: Union[str, None]
    standard_size: str

    ean: Union[int, None]
    myntra_id: Union[int, None]
    ajio_id: Union[int, None]
    ajio_ean: Union[int, None]

    garment_waist: Union[float, None]
    inseam_length: Union[float, None]
    to_fit_waist: Union[float, None]
    across_shoulder: Union[float, None]
    chest: Union[float, None]
    front_length: Union[float, None]
    to_fit_foot_length: Union[float, None]
    shoe_weight: Union[str, None]
    outseam_length: Union[float, None]
    bust: Union[float, None]
    to_fit_bust: Union[float, None]
    hip: Union[float, None]

    # deleted: Union[bool, None] = False


class EditSize(BaseModel):
    barcode_size: Union[str, None]
    standard_size: str

    ean: Union[int, None]
    myntra_id: Union[int, None]
    ajio_id: Union[int, None]
    ajio_ean: Union[int, None]

    garment_waist: Union[float, None]
    inseam_length: Union[float, None]
    to_fit_waist: Union[float, None]
    across_shoulder: Union[float, None]
    chest: Union[float, None]
    front_length: Union[float, None]
    to_fit_foot_length: Union[float, None]
    shoe_weight: Union[str, None]
    outseam_length: Union[float, None]
    bust: Union[float, None]
    to_fit_bust: Union[float, None]
    hip: Union[float, None]


class CreateProduct(BaseModel):
    style_id: str
    hierarchy: Hierarchy
    sizes: list[Size]
    story: str
    colour: Colour
    mrp: Union[int, None]
    cost: Union[int, None]
    tags: Union[list[str], None]
    season: Union[str, None]
    exclusive: str
    design: Design
    fabric: Fabric
    dates: Union[Dates, None]
    product_description: Union[str, None]  # Add this line

class EditProduct(BaseModel):
    hierarchy: Union[Hierarchy, None]
    sizes: Union[list[EditSize], None]
    story: Union[str, None]
    colour: Union[EditColour, None]
    mrp: Union[int, None]
    cost: Union[int, None]
    tags: Union[list[str], None]
    season: Union[str, None]
    exclusive: Union[str, None]
    design: Union[Design, None]
    fabric: Union[Fabric, None]
    dates: Union[Dates, None]
    status: Union[str, None]
    deleted: Union[bool, None]
    style_id: Union[str, None]  # Add this line
    product_description: Union[str, None]  # Add this line
    

class ReadProduct(CreateProduct):
    fabric: ReadFabric
    product_id: str
    deleted: bool
    status: str
    images: Union[list[str], None]
    created_by: str
    updated_by: Union[str, None]
    created_at: datetime.datetime
    updated_at: Union[datetime.datetime, None]


class ProductSummary(BaseModel):
    gender: str
    usage: str
    product: str
    sub_product: str
    fit: str
    colour_family: str
    mrp: Union[int, None]
    cost: Union[int, None]
    style_id: str
    product_id: str
    status: str
    exclusive: str
    first_live_date: Union[datetime.date, None]
    sizes: Union[list[str], None]
    image: Union[str, None]
    product_description: Union[str, None]  # Add this line


class BulkUpdateStatusRequest(BaseModel):
    productIds: List[str]
    newStatus: str

    
