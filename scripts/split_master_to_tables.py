import pandas as pd

df = pd.read_csv("data/master.csv")

sizes = []
styles = []
unique_products = set()

for index, row in df.iterrows():
    if row["product_id"] not in unique_products:
        unique_products.add(row["product_id"])
        styles.append(
            {
                "product_id": row["product_id"],
                "style_id": row["style_id"],
                "gender": row["gender"],
                "vertical": row["vertical"],
                "fabric_category": row["fabric_category"],
                "brand": row["brand"],
                "usage": row["usage"],
                "brick": row["brick"],
                "product": row["product"],
                "sub_product": row["sub_product"],
                "look": row["look"],
                "target_audience": row["target_audience"],
                "fit": row["fit"],
                "mrp": row["mrp"],
                "cost": row["cost"],
                "colour_family": row["colour_family"],
                "primary_colour": row["primary_colour"],
                "secondary_colour": row["secondary_colour"],
                "tertiary_colour": row["tertiary_colour"],
                "tags": row["tags"],
                "season": row["season"],
                "exclusive": row["exclusive"],
                "garment_pattern": row["garment_pattern"],
                "print_pattern_type": row["print_pattern_type"],
                "number_of_components": row["number_of_components"],
                "number_of_pockets": row["number_of_pockets"],
                "pocket_type": row["pocket_type"],
                "length": row["length"],
                "neck": row["neck"],
                "collar": row["collar"],
                "placket": row["placket"],
                "sleeve_length": row["sleeve_length"],
                "sleeve_type": row["sleeve_type"],
                "hemline": row["hemline"],
                "waist_rise": row["waist_rise"],
                "closure": row["closure"],
                "footwear_ankle_type": row["footwear_ankle_type"],
                "footwear_insole": row["footwear_insole"],
                "fabric_code": row["fabric_code"],
                "fabric_rate": row["fabric_rate"],
                "fabric_story": row["fabric_story"],
                "fabric_composition": row["fabric_composition"],
                "fabric_hsn": row["fabric_hsn"],
                "fabric_weave_pattern": row["fabric_weave_pattern"],
                "fabric_vendor": row["fabric_vendor"],
                "denim_cast": row["denim_cast"],
                "denim_wash": row["denim_wash"],
                "wash_care": row["wash_care"],
                "footwear_upper_material": row["footwear_upper_material"],
                "footwear_sole_material": row["footwear_sole_material"],
                "first_grn_date": row["first_grn_date"],
                "first_live_date": row["first_live_date"],
                "first_sold_date": row["first_sold_date"],
                "status": row["status"],
                "created_at": row["created_at"],
                "created_by": row["created_by"],
                "updated_at": row["updated_at"],
                "updated_by": row["updated_by"],
                "images": row["images"],
                "deleted": row["deleted"],
            }
        )
    sizes.append(
        {
            "ean": row["ean"],
            "myntra_id": row["myntra_id"],
            "ajio_id": row["ajio_id"],
            "ajio_ean": row["ajio_ean"],
            "product_id": row["product_id"],
            "standard_size": row["standard_size"],
            "barcode_size": row["barcode_size"],
            "garment_waist": row["garment_waist"],
            "inseam_length": row["inseam_length"],
            "to_fit_waist": row["to_fit_waist"],
            "across_shoulder": row["across_shoulder"],
            "chest": row["chest"],
            "front_length": row["front_length"],
            "to_fit_foot_length": row["to_fit_foot_length"],
            "shoe_weight": row["shoe_weight"],
        }
    )
styles = pd.DataFrame.from_records(styles)
sizes = pd.DataFrame.from_records(sizes)
styles.to_csv("data/styles.csv", index=False)
sizes.to_csv("data/sizes.csv", index=False)
