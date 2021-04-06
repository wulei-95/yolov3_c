// Generated by gencpp from file swd_msgs/Camera.msg
// DO NOT EDIT!


#ifndef SWD_MSGS_MESSAGE_CAMERA_H
#define SWD_MSGS_MESSAGE_CAMERA_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace swd_msgs
{
template <class ContainerAllocator>
struct Camera_
{
  typedef Camera_<ContainerAllocator> Type;

  Camera_()
    : img()  {
    }
  Camera_(const ContainerAllocator& _alloc)
    : img(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<uint8_t, typename ContainerAllocator::template rebind<uint8_t>::other >  _img_type;
  _img_type img;





  typedef boost::shared_ptr< ::swd_msgs::Camera_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::swd_msgs::Camera_<ContainerAllocator> const> ConstPtr;

}; // struct Camera_

typedef ::swd_msgs::Camera_<std::allocator<void> > Camera;

typedef boost::shared_ptr< ::swd_msgs::Camera > CameraPtr;
typedef boost::shared_ptr< ::swd_msgs::Camera const> CameraConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::swd_msgs::Camera_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::swd_msgs::Camera_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace swd_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'swd_msgs': ['/home/xu/ROS_P/rostest_ws/src/swd_msgs/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::swd_msgs::Camera_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::swd_msgs::Camera_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::swd_msgs::Camera_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::swd_msgs::Camera_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::swd_msgs::Camera_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::swd_msgs::Camera_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::swd_msgs::Camera_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d0ca8db996a182c8c227c4286d7dae7b";
  }

  static const char* value(const ::swd_msgs::Camera_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd0ca8db996a182c8ULL;
  static const uint64_t static_value2 = 0xc227c4286d7dae7bULL;
};

template<class ContainerAllocator>
struct DataType< ::swd_msgs::Camera_<ContainerAllocator> >
{
  static const char* value()
  {
    return "swd_msgs/Camera";
  }

  static const char* value(const ::swd_msgs::Camera_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::swd_msgs::Camera_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8[] img\n\
";
  }

  static const char* value(const ::swd_msgs::Camera_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::swd_msgs::Camera_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.img);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Camera_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::swd_msgs::Camera_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::swd_msgs::Camera_<ContainerAllocator>& v)
  {
    s << indent << "img[]" << std::endl;
    for (size_t i = 0; i < v.img.size(); ++i)
    {
      s << indent << "  img[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.img[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // SWD_MSGS_MESSAGE_CAMERA_H