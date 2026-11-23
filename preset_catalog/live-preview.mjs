for (const card of document.querySelectorAll(".card[data-rank]")) {
  const thumbnail = card.querySelector("a");
  const name = card.dataset.name;
  const model = new URL(`models/${encodeURIComponent(name)}.xml`, document.baseURI);
  const viewer = new URL("https://live.mujoco.org/");
  viewer.searchParams.set("model", model.href);
  thumbnail.href = viewer.href;
  thumbnail.target = "_blank";
  thumbnail.rel = "noopener";
  thumbnail.setAttribute("aria-label", `Simulate ${name} in MuJoCo Live (opens in a new tab)`);
}
