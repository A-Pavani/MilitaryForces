<?php

include("./incl/haad.php");
include("./include/function.php");

if(isset($_REQUEST['submit'])){
    include("./incl/db_conn.php");
    $missilename=$_REQUEST['missilename'];
    $missileid=$_REQUEST['missileid'];  
    $missiletype=$_REQUEST['missiletype'];
    $missilestatus=$_REQUEST['missilestatus'];
   
    $sql="INSERT INTO equipment_table(missilename,missileid,missiletype,missilestatus)
    VALUES('$missilename','$missileid','$missiletype','$missilestatus')";
    if(mysqli_query($conn,$sql)){
        my_alert();
    }else{
        echo "Error :" .$sql."<br>".mysqli_error($conn);
    }
    mysqli_close($conn);
}
?>
  <div class="icon">
        <h2 class="logo">Military Equipments</h2>
    </div> 
    <form   method="POST">
    <!-- <fieldset> -->
    <div class="container">
    Enter Missile name : <input type="text" name="missilename" placeholder="Enter missile name" class="box-1"><br/>
    Enter Missile ID : <input type="number" name="missileid" placeholder="Enter missile ID" class="box-2"><br/>
    Enter Missile type : <input type="text" name="missiletype" placeholder="Enter missile type" class="box-3"><br/>
    Enter Missile status : <input type="text"  name="missilestatus" placeholder="Enter status of the missile" class="box-4"><br/>
    <button class="btnn"  type="submit" name="submit" >Submit</button>
    <button class="btnn"><a href="combinedpage.php">Back</a></button>
    </div>
    <!-- </fieldset> -->
    </form>


<?php

include("./incl/faat.php");

?>