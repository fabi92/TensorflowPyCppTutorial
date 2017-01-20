#include <fstream>
#include <sstream>
#include <random>
#include <unordered_map>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/features2d.hpp>

#include <boost/filesystem.hpp>


#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tbb/tbb.h>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

#include <boost/filesystem.hpp>

#define SQR(x) ((x)*(x))

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;

int feat_dim=128;
tensorflow::Session* session;

Mat showRGBDPatch(Mat &patch, bool show=true)
{
    vector<Mat> channels;
    cv::split(patch,channels);
    Mat RGB,D;
    cv::merge(vector<Mat>({channels[0],channels[1],channels[2]}),RGB);
    Mat out(patch.rows,patch.cols*2,CV_32FC3);
    RGB.copyTo(out(Rect(0,0,patch.cols,patch.rows)));
    if (channels.size()>3)
    {
        cv::merge(vector<Mat>({channels[3],channels[3],channels[3]}),D);
        D.copyTo(out(Rect(patch.cols,0,patch.cols,patch.rows)));
    }

    cv::resize(out,out,Size(),4,4);

    if(show) {imshow("R G B D",out); waitKey();}
    return out;
}

vector<Mat> regressFeaturesTensorFlow(vector<Mat> &samples, bool showRecons)
{
    if (samples.empty()) throw runtime_error("Empty samples!");

    // Figure out dimensions
    size_t batch_size = 1000;
    int width = samples[0].size().width;
    int height = samples[0].size().height;
    int depth = samples[0].channels();
    int bytes_per_sample_in = width*height*depth*sizeof(float);
    int bytes_per_sample_out = feat_dim*sizeof(float);


    // Build 4-dim Tensor for input and get Eigen mapping
    tensorflow::TensorShape shape({batch_size, height, width, depth});
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, shape);
    vector<pair<string, tensorflow::Tensor> > input({ {"x",tensor} });
    auto mapping_in = tensor.tensor<float, 4>();

    // Define input/output tensors
    vector< string > labels( {"z"} );
    if (showRecons) labels.push_back("y");
    vector<tensorflow::Tensor> cae_out;

    StopWatch timer;

    vector<Mat> feats;

    size_t nr_batches = samples.size() / batch_size + 1;
    for (size_t batch=0; batch < nr_batches; ++batch)
    {

        size_t base = batch*batch_size;
        if (base > samples.size()) continue;

        int samplesToProcess = std::min(samples.size()-base, batch_size);
        for (int s = 0; s < samplesToProcess; s++){
            memcpy(&mapping_in(s,0,0,0), samples[base+s].data, bytes_per_sample_in);
        }

        tensorflow::Status status = session->Run(input, labels, {}, &cae_out);
        if(!status.ok()) throw runtime_error("Could not run graph: " + status.error_message());


        auto mapping_out = cae_out[0].tensor<float,2>();   // Get 2-dim Eigen tensor mapping for output

        for (int s = 0; s < samplesToProcess; s++){
            Mat m(1, feat_dim, CV_32F);
            memcpy(m.data, &mapping_out(s,0), bytes_per_sample_out);
            feats.push_back(m);
        }

        if (showRecons)
        {
            auto mapping_recon = cae_out[1].tensor<float,4>();   // Get 2-dim Eigen tensor mapping for output
            for (int s = 0; s < samplesToProcess; s++)
            {
                Mat r(64,64, CV_32FC(depth), &mapping_recon(s,0,0,0));
                memcpy(r.data, &mapping_recon(s,0,0,0), bytes_per_sample_in);
                showRGBDPatch(r,true);
            }
        }

    }
    float elaps = timer.elapsedMs();
    cerr << " Time to run: " << elaps << endl;

    return feats;

}

int main(int argc, char *argv[])
{
    const string network = "/models/cae_"+to_string(feat_dim)+".pb";


    tensorflow::SessionOptions opts;
    tensorflow::GPUOptions *gpu_opts = new tensorflow::GPUOptions();

    opts.config.set_allow_soft_placement(false); // Allow mixing CPU and GPU data
    gpu_opts.set_allow_growth(true);            // Don't hog the full GPU at once
    opts.config.set_allocated_gpu_options(gpu_opts);

    tensorflow::Status status = tensorflow::NewSession(opts, &session);
    if(!status.ok()) cerr << "Could not creat new sessions: " << status.error_message() << endl;

    tensorflow::GraphDef graph_def;
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), network, &graph_def);
    if(!status.ok()) cerr << "Could not read Proto File: " << status.error_message() << endl;

    status = session->Create(graph_def);
    if(!status.ok()) cerr << "Could not create graph " << status.error_message() << endl;


    string rgb_1 = "path/rgb_1.png";
    string dep_1 = "path/dep_1.png";
    Mat m_col_1 = imread(rgb_1);
    Mat m_dep_1 = imread(dep_1,-1);

    string rgb_2 = "path/rgb_2.png";
    string dep_2 = "path/dep_2.png";
    Mat m_col_2 = imread(rgb_2);
    Mat m_dep_2 = imread(dep_2,-1);

    m_col_1.convertTo(m_col_1, CV_32FC3, 1.f/255.f);
    m_col_2.convertTo(m_col_2, CV_32FC3, 1.f/255.f);

    cv::resize(m_col_1, m_col_1, cv::Size(64,64),0,0);
    cv::resize(m_dep_1, m_dep_1, cv::Size(64,64),0,0, INTER_NEAREST);
    cv::resize(m_col_2, m_col_2, cv::Size(64,64),0,0);
    cv::resize(m_dep_2, m_dep_2, cv::Size(64,64),0,0, INTER_NEAREST);

    Mat patch_1, patch_2;
    cv::merge(vector<Mat>({m_col_1, m_dep_1}), patch_1);
    cv::merge(vector<Mat>({m_col_2, m_dep_2}), patch_2);


    vector< Mat> samples({patch_1, patch_2});

    cout << "loaded. running regression of features..." << endl;
    vector<Mat> feats = regressFeaturesTensorFlow(samples, true);

    showRGBDPatch(patch_1);

    session->Close();
    return 0;
}