from data_import import get_tracks

tracks = get_tracks("datasets/emotify/Separated_and_mixed_versions","datasets/emotify/emotify_data.csv",
                    sources=["Instrumental", "Mixed", "Vocals"], amount_to_take=2)
for track in tracks:
    print(track.label)
    print(track.sound)
    print(track.source)