import boto3

test_cols = {
    "date": "S", 
    "team_name": "S", 
    "p_name": "S",
    "p_rating": "N",
    "p_passes": "N"
}

attrs = []

for key in test_cols:
    attrs.append(
        {
            "AttributeName": key,
            "AttributeType": test_cols[key]
        }
    )
    
test_df = df.loc[df.matchid == "328136"][test_cols]

dynamodb = boto3.resource('dynamodb')

# table = dynamodb.create_table(
#     TableName='pl_football',
#     KeySchema=[
#         {
#             'AttributeName': 'p_name',
#             'KeyType': 'HASH'
#         }
#     ],
#     AttributeDefinitions=[
#         {
#             'AttributeName': 'p_name',
#             'AttributeType': 'S'
#         }
#     ],
#     ProvisionedThroughput={
#         'ReadCapacityUnits': 5,
#         'WriteCapacityUnits': 5
#     }
# )

table = dynamodb.Table("pl_football")
test_cols = ["p_name", "goals", "p_passes", "team_name"]

for idx in np.arange(10):
    table.put_item(
        Item = {
            "p_name": df.iloc[idx]["p_name"],
            "p_passes": df.iloc[idx]["p_passes"],
            "team_name": df.iloc[idx]["team_name"]
        }
    )



pd.DataFrame(ret)