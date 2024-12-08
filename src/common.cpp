#include "common.h"
#include <iostream>
#include <random>
#include "toml.hpp"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

#define clamp(x, min, max) (x) < (max) ? ((x) > (min) ? (x) : (min)) : (max)


Config::Config(const std::string toml_filepath)
    : trim(false), subsample(1.0f), mse_threshold(1e-5f), resize(1.0f),
    io{ "", "", "", "" }
{
    std::string base_filename = toml_filepath.substr(toml_filepath.find_last_of("/\\") + 1);
    Logger(LogLevel::Info) << "Reading configurations from " << base_filename;
    parse_toml(toml_filepath);
}

void Config::parse_toml(const std::string toml_filepath)
{
    toml::table tbl;

    try
    {
        tbl = toml::parse_file(toml_filepath);
    }
    catch (const toml::parse_error& err)
    {
        std::string err_msg = "Error parsing file '";
        err_msg += *err.source().path;  // Dereference the pointer to get the path as a string
        err_msg += "': ";
        err_msg += err.description();
        err_msg += "\n";

        throw std::runtime_error(err_msg);
    }

    std::optional<std::string> desc = tbl["info"]["description"].value<std::string>();
    Logger(LogLevel::Info) << desc.value();

    // Parse IO section
    if (tbl.contains("io"))
    {
        auto io_section = tbl["io"];
        io.target = io_section["target"].value_or("");
        io.source = io_section["source"].value_or("");
        io.output = io_section["output"].value_or("");
        io.visualization = io_section["visualization"].value_or("");
    }

    // Parse parameters
    if (tbl.contains("params"))
    {
        auto params_section = tbl["params"];
        mode = params_section["mode"].value_or(1);
        trim = params_section["trim"].value_or(false);
        subsample = params_section["subsample"].value_or(1.0f);
        mse_threshold = params_section["mse_threshold"].value_or(1e-5f);
        resize = params_section["resize"].value_or(1.0f);

        // Check bounding conditions
        subsample = clamp(subsample, 0.0f, 1.0f);
        mse_threshold = clamp(mse_threshold, 1e-10f, INFINITY);
    }

    Logger(LogLevel::Info) << "Config parsed successfully!";
}

size_t load_cloud_ply(const std::string& ply_filepath, const float& subsample, const float& resize, std::vector<glm::vec3>& cloud)
{
    size_t num_points = 0;

    try
    {
        std::ifstream file_stream(ply_filepath, std::ios::binary);
        if (!file_stream)
        {
            throw std::runtime_error("Unable to open file: " + ply_filepath);
        }

        tinyply::PlyFile ply_file;
        ply_file.parse_header(file_stream);

        std::shared_ptr<tinyply::PlyData> vertices;

        try
        {
            vertices = ply_file.request_properties_from_element("vertex", { "x", "y", "z" });
        }
        catch (const std::exception& err)
        {
            throw std::runtime_error("PLY file missing 'x', 'y', or 'z' vertex properties.");
        }

        ply_file.read(file_stream);

        if (vertices && vertices->count > 0)
        {
            size_t total_points = vertices->count;
            num_points = static_cast<size_t>(total_points * subsample);
            cloud.reserve(num_points);  // Pre-allocate space for PointCloud

            const float* vertex_buffer = reinterpret_cast<const float*>(vertices->buffer.get());

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0, 1.0);

            size_t index = 0;
            for (size_t i = 0; i < total_points && index < num_points; ++i)
            {
                if (dis(gen) <= subsample)
                {
                    cloud.emplace_back(resize * vertex_buffer[3 * i + 0],
                        resize * vertex_buffer[3 * i + 1],
                        resize * vertex_buffer[3 * i + 2]);
                    ++index;
                }
            }

            // Adjust num_points if fewer points were randomly selected
            num_points = index;
        }
        else
        {
            throw std::runtime_error("No vertices found in the PLY file.");
        }
    }
    catch (const std::exception& err)
    {
        throw std::runtime_error(std::string("Error reading PLY file: ") + err.what());
    }

    Logger(LogLevel::Info) << "Point cloud " << ply_filepath << " loaded with " << num_points << " points!";
    return num_points;
}

size_t load_cloud_txt(const std::string& txt_filepath, const float& subsample, const float& resize, std::vector<glm::vec3>& cloud)
{
    size_t num_points = 0;

    try
    {
        std::ifstream file_stream(txt_filepath);
        if (!file_stream.is_open())
        {
            throw std::runtime_error("Unable to open TXT file: " + txt_filepath);
        }

        int total_points = 0;
        file_stream >> total_points;

        if (total_points <= 0)
        {
            throw std::runtime_error("Invalid number of points in the TXT file: " + txt_filepath);
        }

        num_points = static_cast<size_t>(total_points * subsample);
        cloud.reserve(num_points);  // Pre-allocate space for the point cloud

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        size_t index = 0;
        for (int i = 0; i < total_points; ++i)
        {
            float x, y, z;
            if (!(file_stream >> x >> y >> z))
            {
                throw std::runtime_error("Error reading point data from TXT file: " + txt_filepath);
            }

            if (dis(gen) <= subsample && index < num_points)
            {
                cloud.emplace_back(resize * x, resize * y, resize * z);
                ++index;
            }
        }

        // Adjust num_points if fewer points were randomly selected
        num_points = index;

        file_stream.close();
    }
    catch (const std::exception& err)
    {
        throw std::runtime_error(std::string("Error reading TXT file: ") + err.what());
    }

    Logger(LogLevel::Info) << "Point cloud "<< txt_filepath << " loaded with " << num_points << " points!";
    return num_points;
}

size_t load_cloud(const std::string& filepath, const float& subsample, const float& resize, std::vector<glm::vec3>& cloud)
{
    auto dot_pos = filepath.find_last_of('.');
    if (dot_pos == std::string::npos)
    {
        throw std::runtime_error("Filepath does not have a valid extension: " + filepath);
    }

    std::string extension = filepath.substr(dot_pos + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == "ply")
    {
        return load_cloud_ply(filepath, subsample, resize, cloud);
    }
    else if (extension == "txt")
    {
        return load_cloud_txt(filepath, subsample, resize, cloud);
    }
    else
    {
        throw std::runtime_error("Unsupported file extension: " + extension);
    }
}