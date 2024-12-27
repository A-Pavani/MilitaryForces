<?php
$servername="localhost";
$username="root";
$password="";
$db_name="soldier_details";
$conn=mysqli_connect($servername,$username,$password,$db_name);
if(!$conn){
    die("connection failed :".mysqli_connect_error());

}
// echo "connected successfuly";


?>