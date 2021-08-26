#include <string>

#include <gpd/candidate/candidates_generator.h>
#include <gpd/candidate/hand.h>
#include <gpd/candidate/hand_geometry.h>
#include <gpd/descriptor/image_12_channels_strategy.h>
#include <gpd/descriptor/image_15_channels_strategy.h>
#include <gpd/descriptor/image_1_channels_strategy.h>
#include <gpd/descriptor/image_3_channels_strategy.h>
#include <gpd/descriptor/image_generator.h>
#include <gpd/net/classifier.h>
#include <gpd/util/config_file.h>
#include <gpd/util/plot.h>

namespace gpd {
namespace test {
namespace {

int DoMain(int argc, char *argv[]) {

  // View point from which the camera sees the point cloud.
  Eigen::Matrix3Xd view_points(3, 1);
  view_points.setZero();

  // Load point cloud from file
  std::string filename = argv[1];
  util::Cloud cloud(filename, view_points);
  if (cloud.getCloudOriginal()->size() == 0) {
    std::cout << "Error: Input point cloud is empty or does not exist!\n";
    return (-1);
  }

  double normals_radius = 0.03;

  // Calculate surface normals.
  cloud.calculateNormals(4, normals_radius);
  cloud.setNormals(cloud.getNormals() * (-1.0));

//  std::string normal_path="/home/wuxr/Datasets_graspnet/";
// cloud.writeNormalsToFile(normal_path,cloud.getNormals());
 std::cout<<cloud.getNormals() ;
  return 0;
}

}  // namespace
}  // namespace test
}  // namespace gpd

int main(int argc, char *argv[]) { return gpd::test::DoMain(argc, argv); }
