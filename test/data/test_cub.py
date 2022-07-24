from scarce_shot_learn.data.cub import load_cub, load_cub_images


def test_cub_image_data():

    cub_data = load_cub_images((64, 64))

    cub_train_data = cub_data["train"][0]
    cub_train_labels = cub_data["train"][1]

    cub_test_data = cub_data["test"][0]
    cub_test_labels = cub_data["test"][1]

    assert cub_train_data.shape == (3000, 64, 64, 3)
    assert cub_train_labels.shape == (3000,)

    assert cub_test_data.shape == (3033, 64, 64, 3)
    assert cub_test_labels.shape == (3033,)
