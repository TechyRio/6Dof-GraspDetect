#include <gpd/candidate/frame_estimator.h>

namespace gpd {
namespace candidate {

std::vector<LocalFrame> FrameEstimator::calculateLocalFrames(
    const util::Cloud &cloud_cam, const std::vector<int> &indices,
    double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA> &kdtree) const {
  double t1 = omp_get_wtime();
  std::vector<std::unique_ptr<LocalFrame>> frames;
  frames.resize(indices.size());

#ifdef _OPENMP  // parallelization using OpenMP
#pragma omp parallel for num_threads(num_threads_)
#endif
  for (int i = 0; i < indices.size(); i++) {
    const pcl::PointXYZRGBA &sample =
        cloud_cam.getCloudProcessed()->points[indices[i]];
    frames[i] =
        calculateFrame(cloud_cam.getNormals(),
                       sample.getVector3fMap().cast<double>(), radius, kdtree);
  }

  std::vector<LocalFrame> frames_out;
  for (int i = 0; i < frames.size(); i++) {
    if (frames[i]) {
      frames_out.push_back(*frames[i].get());
    }
  }

  double t2 = omp_get_wtime();
  printf("Estimated %zu frames in %3.4fs.\n", frames_out.size(), t2 - t1);

  return frames_out;
}

std::vector<LocalFrame> FrameEstimator::calculateLocalFrames(
    const util::Cloud &cloud_cam, const Eigen::Matrix3Xd &samples,
    double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA> &kdtree) const {
  double t1 = omp_get_wtime();
  std::vector<std::unique_ptr<LocalFrame>> frames;
  frames.resize(samples.cols());

#ifdef _OPENMP  // parallelization using OpenMP
#pragma omp parallel for num_threads(num_threads_)
#endif
  for (int i = 0; i < samples.cols(); i++) {
    frames[i] =
        calculateFrame(cloud_cam.getNormals(), samples.col(i), radius, kdtree);                                                              //从众多nomals中找到坐标系。（法向量、采样点、距离、kd）
  }

  // Only keep frames that are not null.
  std::vector<LocalFrame> frames_out;
  for (int i = 0; i < frames.size(); i++) {
    if (frames[i]) {
      frames_out.push_back(*frames[i].get());                 //返回指针。
    }
  }

  double t2 = omp_get_wtime();
  printf("Estimated %zu frames in %3.4fs.\n", frames_out.size(), t2 - t1);

  return frames_out;
}

std::unique_ptr<LocalFrame> FrameEstimator::calculateFrame(
    const Eigen::Matrix3Xd &normals, const Eigen::Vector3d &sample,
    double radius, const pcl::KdTreeFLANN<pcl::PointXYZRGBA> &kdtree) const {
  std::unique_ptr<LocalFrame> frame = nullptr;
  std::vector<int> nn_indices;
  std::vector<float> nn_dists;
  pcl::PointXYZRGBA sample_pcl = eigenVectorToPcl(sample);

  if (kdtree.radiusSearch(sample_pcl, radius, nn_indices, nn_dists) > 0) {                //使用KD树，取限定范围内的采样点。对符合的采样进行
    Eigen::Matrix3Xd nn_normals(3, nn_indices.size());                                                 //3*nn_indices的一个矩阵。indices表示一个索引。

    for (int i = 0; i < nn_normals.cols(); i++) {
      nn_normals.col(i) = normals.col(nn_indices[i]);
    }

    frame = std::make_unique<LocalFrame>(sample);
    frame->findAverageNormalAxis(nn_normals);                                   //从众多nn_nomals中找到坐标系
  }

  return frame;
}

pcl::PointXYZRGBA FrameEstimator::eigenVectorToPcl(
    const Eigen::Vector3d &v) const {
  pcl::PointXYZRGBA p;
  p.x = v(0);
  p.y = v(1);
  p.z = v(2);
  return p;
}

}  // namespace candidate
}  // namespace gpd
